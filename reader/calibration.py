"""
calibration.py — Detecção automática do grid de bolhas.

Encontra as posições X e Y das bolhas na imagem corrigida extraindo
os centroides de todas as bolhas, agrupando-os e aplicando uma
transformação (Homografia) para estimar a posição exata de cada
bolha individual, corrigindo pequenas rotações ou distorções.
"""

import cv2
import numpy as np


def _find_dense_clusters(values: list[int], expected_clusters: int, min_cluster_size: float, min_dist: float) -> list[int]:
    """
    Encontra clusters densos em uma lista de coordenadas 1D (KDE simplificado).
    """
    if not values:
        return []
        
    bandwidth = max(5.0, min_dist / 3.0)
    # Adiciona padding no grid para não cortar os picos (gaussiana) que caem nas extremidades
    pad = int(bandwidth * 3)
    grid = np.arange(min(values) - pad, max(values) + pad + 1)
    density = np.zeros_like(grid, dtype=np.float32)
    
    for v in values:
        density += np.exp(-0.5 * ((grid - v) / bandwidth)**2)
        
    peaks = []
    for i in range(1, len(density) - 1):
        if density[i] > density[i - 1] and density[i] > density[i + 1]:
            peaks.append((grid[i], density[i]))
            
    peaks.sort(key=lambda p: p[1], reverse=True)
    
    final_peaks = []
    for p in peaks:
        if p[1] < min_cluster_size:
            continue
        if all(abs(p[0] - fp) > min_dist for fp in final_peaks):
            final_peaks.append(p[0])
            if len(final_peaks) == expected_clusters:
                break
                
    if len(final_peaks) < expected_clusters:
        return []
        
    return sorted(final_peaks)


def auto_calibrate_grid(warped_gray: np.ndarray, expected_radius: int) -> tuple[dict, dict, dict] | None:
    """
    Detecta o grid de bolhas 2D.
    
    Returns:
        (x_positions, y_positions, grid_2d)
        Onde grid_2d tem o formato: {"1": {"A": (cx, cy), ...}, ...}
    """
    binary = cv2.adaptiveThreshold(
        warped_gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=31, C=10
    )
    
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    expected_area = np.pi * (expected_radius ** 2)
    min_area = expected_area * 0.2
    max_area = expected_area * 3.0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(max(h, 1))
            if 0.4 < aspect < 2.5:
                # Filtragem rigorosa por circularidade para ignorar os retângulos com os números das questões
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.80:
                        # Usa fitEllipse para obter o centroide com precisão sub-pixel
                        if len(cnt) >= 5:
                            ellipse = cv2.fitEllipse(cnt)
                            centroids.append((ellipse, cnt))
                    
    if len(centroids) < 15:
        return None
        
    xs = [c[0][0][0] for c in centroids]
    ys = [c[0][0][1] for c in centroids]
    
    min_dist = expected_radius * 1.5
    
    x_centers = _find_dense_clusters(xs, expected_clusters=5, min_cluster_size=2.0, min_dist=min_dist)
    y_centers = _find_dense_clusters(ys, expected_clusters=10, min_cluster_size=1.0, min_dist=min_dist)
    
    if not x_centers or not y_centers:
        return None
        
    # Associação: Para cada centroide real, acha o ponto (ideal_x, ideal_y) ortogonal mais próximo
    src_pts = []
    dst_pts = []
    
    for ellipse_data, cnt in centroids:
        cx, cy = ellipse_data[0]
        # Encontra a coluna (X) mais próxima
        closest_x = min(x_centers, key=lambda ideal: abs(ideal - cx))
        # Encontra a linha (Y) mais próxima
        closest_y = min(y_centers, key=lambda ideal: abs(ideal - cy))
        
        # Só associa se estiver a uma distância razoável
        if abs(closest_x - cx) < min_dist and abs(closest_y - cy) < min_dist:
            src_pts.append([closest_x, closest_y])
            dst_pts.append([cx, cy])
            
    if len(src_pts) < 10:
        return None
        
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    
    # Calcula uma homografia (ou transformação afim) do grid ideal para os centroides reais
    # Usamos RANSAC para ignorar contornos que não eram bolhas
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return None
        
    alts = ["A", "B", "C", "D", "E"]
    x_positions = {alt: x for alt, x in zip(alts, x_centers)}
    y_positions = {str(i + 1): y for i, y in enumerate(y_centers)}
    
    grid_2d = {}
    
    # Mapeia as coordenadas ideais pela homografia para obter o 2D exato
    for i, ideal_y in enumerate(y_centers):
        q_str = str(i + 1)
        grid_2d[q_str] = {}
        for j, ideal_x in enumerate(x_centers):
            alt = alts[j]
            pt = np.array([[[ideal_x, ideal_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, H)
            # Para construir o grid com elipses, vamos calcular a largura/altura médias
            # ou usar a homografia para transformar um círculo ideal na elipse esperada.
            # Porém, a matriz H também aplica warp na forma. 
            # Uma aproximação elegante: pegar a elipse detectada real se houver!
            
            # Procura o centroide real mais próximo da coordenada ideal
            real_cx, real_cy = transformed[0][0]
            
            # Encontra a elipse real correspondente se existir
            best_ellipse = None
            min_err = min_dist
            for (ell, _cnt) in centroids:
                err = np.hypot(ell[0][0] - real_cx, ell[0][1] - real_cy)
                if err < min_err:
                    min_err = err
                    best_ellipse = ell
            
            if best_ellipse is not None:
                # Usa o centro exato da bolha detectada para alinhamento sub-pixel, 
                # mas FORÇA o raio matemático perfeito para evitar que contornos 
                # esticados (rasuras, 2 bolhas coladas) invadam outras células.
                exact_cx, exact_cy = best_ellipse[0]
                grid_2d[q_str][alt] = ((exact_cx, exact_cy), (expected_radius*2, expected_radius*2), 0.0)
            else:
                # Fallback: se não detectamos a elipse exata, estimamos o centro projetado
                grid_2d[q_str][alt] = ((int(real_cx), int(real_cy)), (expected_radius*2, expected_radius*2), 0.0)
            
    return x_positions, y_positions, grid_2d
