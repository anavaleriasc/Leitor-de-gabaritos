"""
perspective.py — Detecção do contorno externo do cartão e correção de perspectiva.

Sem marcadores pretos nos cantos, o sistema precisa detectar a borda da folha
pelo contraste com o fundo. Isso exige que a foto tenha bom contraste entre
a folha (branca) e o fundo (escuro ou colorido).

Pipeline:
  1. Encontrar todos os contornos externos
  2. Selecionar o maior contorno que seja um quadrilátero plausível
  3. Ordenar os 4 vértices (TL → TR → BR → BL)
  4. Aplicar getPerspectiveTransform + warpPerspective
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────
# Utilitários de geometria
# ──────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Ordena 4 pontos no sentido: [topo-esquerda, topo-direita, baixo-direita, baixo-esquerda].
    Utiliza ordenação angular (arctan2) pelo centro de massa para garantir precisão
    mesmo quando a folha está rotacionada em ângulos extremos (losangos).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Calcula o centro de massa
    center = np.mean(pts, axis=0)
    
    # Calcula o ângulo de cada ponto em relação ao centro
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # Ordena os pontos pelo ângulo (do menor para o maior)
    # A ordem angular no OpenCV (y cresce para baixo) é:
    # TL (-135°), TR (-45°), BR (45°), BL (135°)
    sorted_pts = pts[np.argsort(angles)]
    
    rect[0] = sorted_pts[0] # TL
    rect[1] = sorted_pts[1] # TR
    rect[2] = sorted_pts[2] # BR
    rect[3] = sorted_pts[3] # BL
    
    # Como as fotos podem ser tiradas em diferentes rotações,
    # garantimos que a borda de cima (TL -> TR) seja a borda curta do papel A4 (largura)
    # e a borda da esquerda (TL -> BL) seja a borda longa (altura).
    dist_top = np.linalg.norm(rect[0] - rect[1])
    dist_left = np.linalg.norm(rect[0] - rect[3])
    
    if dist_top > dist_left:
        # A imagem está deitada (rotação de 90 graus)
        # Rotacionamos os pontos no array para consertar a orientação
        rect = np.roll(rect, -1, axis=0)
        
    return rect


def compute_warped_size(rect: np.ndarray) -> tuple[int, int]:
    """
    Calcula a largura e altura da imagem transformada preservando proporção real.
    Usado apenas como estimativa quando dimensões fixas não são fornecidas.
    """
    (tl, tr, br, bl) = rect

    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bot))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))

    return width, height


# ──────────────────────────────────────────────
# Detecção do contorno externo da folha
# ──────────────────────────────────────────────

def _is_valid_marker(contour: np.ndarray, image_area: float) -> bool:
    """
    Valida se um contorno é um marcador (retângulo preto sólido).
    """
    area = cv2.contourArea(contour)
    # Marcador deve ser grande o suficiente, mas não gigantesco
    if not (image_area * 0.0005 < area < image_area * 0.05):
        return False

    # Deve ser bem sólido (sem buracos grandes)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0 or (area / hull_area) < 0.8:
        return False
        
    # Deve ser grosseiramente retangular
    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / max(h, 1)
    # Os marcadores da imagem parecem retângulos horizontais (ex: 4:1 ou 3:1)
    # Mas vamos ser flexíveis para permitir inclinação (1.5 a 6.0)
    if not (1.5 <= aspect <= 8.0):
        return False
        
    return True


def find_markers(binary: np.ndarray,
                 original_shape: tuple) -> np.ndarray | None:
    """
    Encontra os 4 marcadores nos cantos da imagem.
    
    Returns:
        Array (4, 2) com os centróides dos marcadores ordenados (TL, TR, BR, BL).
    """
    image_area = float(original_shape[0] * original_shape[1])
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_markers = []
    for cnt in contours:
        if _is_valid_marker(cnt, image_area):
            valid_markers.append(cnt)
            
    # Se achamos mais de 4, pegamos os 4 maiores em área
    valid_markers = sorted(valid_markers, key=cv2.contourArea, reverse=True)[:4]
    
    if len(valid_markers) != 4:
        return None
        
    centroids = []
    for cnt in valid_markers:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append([cx, cy])
            
    if len(centroids) != 4:
        return None
        
    return order_points(np.array(centroids, dtype=np.float32))


def _is_valid_quadrilateral(approx: np.ndarray, image_area: float,
                             min_area_ratio: float = 0.10,
                             aspect_min: float = 0.4,
                             aspect_max: float = 2.5) -> bool:
    """
    Valida se um contorno aproximado é um quadrilátero plausível para a folha.
    """
    if len(approx) != 4:
        return False

    area = cv2.contourArea(approx)
    if area < image_area * min_area_ratio:
        return False

    x, y, w, h = cv2.boundingRect(approx)
    aspect = w / max(h, 1)
    if not (aspect_min <= aspect <= aspect_max):
        return False

    if not cv2.isContourConvex(approx):
        return False

    return True


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Aplica Canny edge com limiares calculados automaticamente baseados na mediana."""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


def find_sheet_contour(image: np.ndarray) -> np.ndarray | None:
    """
    Encontra o contorno externo da folha usando redimensionamento e filtro bilateral 
    para extrema robustez contra ruídos, texturas e fundos difíceis.
    """
    original_shape = image.shape[:2]
    image_area = float(original_shape[0] * original_shape[1])
    
    # Redimensiona para uma largura de 800px para ignorar detalhes internos da folha
    ratio = original_shape[0] / 800.0
    res = cv2.resize(image, (int(original_shape[1]/ratio), 800))
    
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = auto_canny(gray)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    epsilon_ratios = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]

    # Área no tamanho reduzido
    res_area = float(res.shape[0] * res.shape[1])

    for contour in contours:
        perimeter = cv2.arcLength(contour, closed=True)
        for eps_ratio in epsilon_ratios:
            approx = cv2.approxPolyDP(contour, eps_ratio * perimeter, closed=True)
            if _is_valid_quadrilateral(approx, res_area):
                # Escala os pontos de volta para a resolução original
                scaled_approx = approx * ratio
                return order_points(scaled_approx.reshape(4, 2).astype(np.float32))

    return None


# ──────────────────────────────────────────────
# Transformação de perspectiva
# ──────────────────────────────────────────────

def warp_perspective(image: np.ndarray,
                     rect: np.ndarray,
                     out_width: int,
                     out_height: int) -> np.ndarray:
    """
    Aplica transformação de perspectiva (homografia) para retificar a imagem do cartão.

    Args:
        image:      imagem original (BGR ou grayscale).
        rect:       4 pontos ordenados [TL, TR, BR, BL].
        out_width:  largura da imagem de saída em pixels.
        out_height: altura da imagem de saída em pixels.

    Returns:
        Imagem retificada com dimensões (out_height, out_width).
    """
    dst = np.array([
        [0,            0],
        [out_width - 1, 0],
        [out_width - 1, out_height - 1],
        [0,            out_height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (out_width, out_height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
    return warped


def detect_and_warp(image: np.ndarray,
                    edges: np.ndarray,
                    out_width: int,
                    out_height: int,
                    use_markers: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Encontra os pontos-chave (folha ou marcadores) e aplica a correção de perspectiva.

    Returns:
        (warped_image, rect) ou (None, None) se nenhum contorno válido for encontrado.
    """
    if use_markers:
        rect = find_markers(edges, image.shape[:2])
    else:
        rect = find_sheet_contour(image)

    if rect is None:
        return None, None

    warped = warp_perspective(image, rect, out_width, out_height)
    return warped, rect


def draw_detected_contour(image: np.ndarray, rect: np.ndarray) -> np.ndarray:
    """Desenha o contorno detectado sobre a imagem para visualização de debug."""
    debug = image.copy()
    pts = rect.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug, [pts], isClosed=True, color=(0, 255, 0), thickness=4)

    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for (x, y), label, color in zip(rect.astype(int), labels, colors):
        cv2.circle(debug, (x, y), 12, color, -1)
        cv2.putText(debug, label, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)
    return debug
