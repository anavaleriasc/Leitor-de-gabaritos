"""
bubbles.py — Segmentação e análise das bolhas do cartão-resposta.

Após a correção de perspectiva, as bolhas estão em posições fixas e conhecidas.
Para cada bolha:
  1. Define-se um ROI (region of interest) quadrado ao redor do centro esperado.
  2. Aplica-se máscara circular interna (ignora bordas da bolha e o anel impresso).
  3. Conta-se a proporção de pixels pretos (=preenchimento pelo lápis/caneta).
  4. Decide-se qual alternativa foi marcada por questão com base em limiares.

Regra de decisão por questão:
  - fill_ratio < min_fill_ratio          → contribuição ignorada
  - alternativa mais marcada > limiar    → essa alternativa é a resposta
  - duas alternativas têm fill_ratio com diferença < ambiguity_margin → AMBIGUA
  - nenhuma passa do limiar             → EM_BRANCO
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field


ALTERNATIVES = ["A", "B", "C", "D", "E"]


# ──────────────────────────────────────────────
# Estruturas de dados
# ──────────────────────────────────────────────

@dataclass
class BubbleResult:
    """Resultado da leitura de uma única bolha."""
    question: int
    alternative: str
    cx: int          # centro x na imagem warped
    cy: int          # centro y na imagem warped
    fill_ratio: float


@dataclass
class QuestionResult:
    """Resultado consolidado de uma questão."""
    question: int
    answer: str                      # alternativa, "EM_BRANCO" ou "AMBIGUA"
    fill_ratios: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    is_blank: bool = False
    is_ambiguous: bool = False


# ──────────────────────────────────────────────
# Análise de bolha individual
# ──────────────────────────────────────────────

def compute_bubble_fill(binary: np.ndarray,
                        ellipse: tuple,
                        inner_mask_ratio: float = 0.7) -> float:
    """
    Calcula a proporção de pixels pretos dentro da bolha modelada por elipse.
    """
    # A imagem binarizada de preprocess_warped_for_bubbles retorna tinta preta como 255.
    mask = np.zeros(binary.shape, dtype=np.uint8)
    
    # ellipse = ((cx, cy), (a, b), angle)
    (cx, cy), (a, b), angle = ellipse
    
    # Escala a elipse para pegar apenas o "miolo" (inner mask)
    scaled_axes = (int(a * inner_mask_ratio / 2), int(b * inner_mask_ratio / 2))
    center = (int(cx), int(cy))
    
    cv2.ellipse(mask, center, scaled_axes, angle, 0, 360, 255, -1)

    # Conta pixels da máscara
    total_pixels = cv2.countNonZero(mask)
    if total_pixels == 0:
        return 0.0

    # Bitwise AND entre a imagem binarizada e a máscara
    # Em warped_binary, a tinta é 255 (branco) devido ao THRESH_BINARY_INV
    bubble_pixels = cv2.bitwise_and(binary, binary, mask=mask)
    filled_pixels = cv2.countNonZero(bubble_pixels)
    return filled_pixels / total_pixels


# ──────────────────────────────────────────────
# Análise de todas as bolhas
# ──────────────────────────────────────────────

def analyze_all_bubbles(binary: np.ndarray,
                        config: dict) -> tuple[list[BubbleResult], list[QuestionResult]]:
    """
    Analisa todas as 50 bolhas (10 questões × 5 alternativas).

    Args:
        binary:  imagem binarizada já corrigida perspectivamente.
        config:  dicionário com sheet_config.json.

    Returns:
        (bubble_results, question_results)
    """
    bubble_radius = config["bubble_radius"]
    inner_mask_radius = config["inner_mask_radius"]
    min_fill = config["min_fill_ratio"]
    min_ink = config.get("min_ink_ratio", 0.20)
    x_positions = config["x_positions"]   # {"A": 300, "B": 410, ...}
    y_positions = config["y_positions"]   # {"1": 330, "2": 420, ...}

    bubble_results: list[BubbleResult] = []
    question_results: list[QuestionResult] = []

    # Se o grid 2D exato foi fornecido pela calibração automática (suporta rotação/skew)
    has_2d_grid = "grid" in config
    grid = config.get("grid", {})

    for q_str, default_cy in y_positions.items():
        q_num = int(q_str)
        fills: dict[str, float] = {}

        for alt in ALTERNATIVES:
            if has_2d_grid and q_str in grid and alt in grid[q_str]:
                ellipse = grid[q_str][alt]
                cx, cy = ellipse[0]
            else:
                cx = x_positions[alt]
                cy = default_cy
                ellipse = ((cx, cy), (bubble_radius*2, bubble_radius*2), 0.0)
                
            # Calcula a razão de raio interno (ex: 14/22)
            inner_ratio = inner_mask_radius / max(1, bubble_radius)
            fill = compute_bubble_fill(binary, ellipse, inner_mask_ratio=inner_ratio)
            fills[alt] = fill
            bubble_results.append(BubbleResult(q_num, alt, int(cx), int(cy), fill))

        qr = _decide_question(q_num, fills, min_fill, min_ink)
        question_results.append(qr)

    return bubble_results, question_results


def _decide_question(question: int,
                     fills: dict[str, float],
                     min_fill: float,
                     min_ink: float) -> QuestionResult:
    """
    Nova regra de decisão solicitada:
    1. Verifica se a melhor alternativa tem tinta suficiente (>= min_fill). Se não -> EM_BRANCO
    2. Verifica se a segunda melhor alternativa também tem alguma tinta (>= min_ink). Se sim -> AMBIGUA
    3. Caso contrário -> resposta válida.
    """
    # Ordena alternativas pelo fill_ratio decrescente
    ranked = sorted(fills.items(), key=lambda kv: kv[1], reverse=True)
    best_alt, best_fill = ranked[0]
    second_alt, second_fill = ranked[1]

    # 1. Nenhuma alternativa atingiu o limiar de marcação válida
    if best_fill < min_fill:
        return QuestionResult(
            question=question,
            answer="EM_BRANCO",
            fill_ratios=fills,
            confidence=best_fill,
            is_blank=True,
            is_ambiguous=False,
        )

    # 2. Há uma marcação válida, mas outra alternativa também tem tinta (rasura ou marcação dupla)
    if second_fill >= min_ink:
        return QuestionResult(
            question=question,
            answer="AMBIGUA",
            fill_ratios=fills,
            confidence=best_fill,
            is_blank=False,
            is_ambiguous=True,
        )

    # Resposta clara
    return QuestionResult(
        question=question,
        answer=best_alt,
        fill_ratios=fills,
        confidence=best_fill,
        is_blank=False,
        is_ambiguous=False,
    )
