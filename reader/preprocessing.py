"""
preprocessing.py — Conversão para tons de cinza, correção de iluminação e limiarização.

Técnicas aplicadas:
  - Conversão RGB → Grayscale
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) para corrigir
    variações de iluminação não-uniforme
  - Limiarização adaptativa de Otsu ou adaptativa por bloco como alternativa
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converte imagem BGR para tons de cinza."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Corrige iluminação não-uniforme causada por sombras, flash ou ângulo de câmera.
    O parâmetro clip_limit controla o nível de amplificação de contraste.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def apply_gaussian_blur(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Aplica suavização gaussiana para reduzir ruído antes da detecção de bordas."""
    # kernel_size deve ser ímpar
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def apply_threshold(gray: np.ndarray, method: str = "otsu", block_size: int = 151) -> np.ndarray:
    """
    Aplica limiarização à imagem.

    Args:
        gray:   imagem em tons de cinza.
        method: 'otsu'     — limiarização global de Otsu (boa para iluminação uniforme)
                'adaptive' — limiarização adaptativa por bloco (robusta para variações locais)

    Returns:
        Imagem binarizada (pixels brancos = fundo, pretos = tinta).
    """
    if method == "adaptive":
        # Um block_size gigante evita que objetos sólidos grandes fiquem vazados (hollow effect)
        # O tamanho deve ser maior que o diâmetro da maior bolha preenchida
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=10,
        )
    else:  # otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary


def preprocess_for_contour_detection(image: np.ndarray, use_markers: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline completo de pré-processamento voltado para detecção geométrica.

    Se use_markers=True, foca em detectar retângulos pretos pequenos sobre o papel.
    Se use_markers=False, foca em fundir a folha inteira numa única grande mancha branca (ou bordas contínuas).
    """
    gray = to_grayscale(image)
    enhanced = apply_clahe(gray)
    blurred = apply_gaussian_blur(enhanced, kernel_size=5)

    if use_markers:
        # Detecção das marcações pretas (fiducials) usando threshold adaptativo
        block_size = 51
        binary = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            blockSize=block_size, C=15
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    else:
        # Detecção do "branco" da folha inteira usando Canny clássico
        # A detecção de bordas funciona melhor que threshold negativo dependendo do contraste
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

    return gray, enhanced, edges


def preprocess_warped_for_bubbles(warped_gray: np.ndarray) -> np.ndarray:
    """
    Pré-processamento aplicado à imagem já corrigida perspectivamente,
    voltado para a análise das bolhas.

    Retorna imagem binarizada onde pixels pretos indicam preenchimento.
    """
    enhanced = apply_clahe(warped_gray, clip_limit=3.0, tile_size=8)
    blurred = apply_gaussian_blur(enhanced, kernel_size=3)
    # block_size de 151 garante que uma bolha sólida não fique "vazada" por dentro
    binary = apply_threshold(blurred, method="adaptive", block_size=151)
    return binary
