"""
visualization.py — Geração de imagens intermediárias de debug.

Produz as 8 imagens de debug salvas em data/debug/ e usadas no relatório.
"""

import cv2
import numpy as np
from reader.bubbles import BubbleResult, QuestionResult, ALTERNATIVES


# ──────────────────────────────────────────────
# Paleta de cores (BGR)
# ──────────────────────────────────────────────
COLOR_MARKED = (0, 200, 0)       # verde — alternativa marcada corretamente
COLOR_BLANK = (200, 200, 0)      # amarelo — questão em branco
COLOR_AMBIGUOUS = (0, 0, 220)    # vermelho — questão ambígua
COLOR_UNMARKED = (180, 180, 180) # cinza — bolha não marcada
COLOR_TEXT = (0, 0, 0)           # preto — texto nos rótulos
COLOR_TEXT_LIGHT = (255, 255, 255)


# ──────────────────────────────────────────────
# Funções auxiliares
# ──────────────────────────────────────────────

def _put_text_centered(img: np.ndarray, text: str, cx: int, cy: int,
                       color: tuple, font_scale: float = 0.5,
                       thickness: int = 1) -> None:
    """Escreve texto centralizado em (cx, cy)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2),
                font, font_scale, color, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Imagem principal de debug das bolhas
# ──────────────────────────────────────────────

def draw_bubbles_debug(warped_bgr: np.ndarray,
                       bubble_results: list[BubbleResult],
                       question_results: list[QuestionResult],
                       config: dict) -> np.ndarray:
    """
    Gera a imagem 08_bubbles_debug.jpg.

    Sobre cada bolha:
      - Círculo externo (raio = bubble_radius) indicando região analisada.
      - Círculo interno (raio = inner_mask_radius) em verde/amarelo/vermelho/cinza.
      - Fill ratio exibido como texto.

    Legenda:
      - Verde:   alternativa marcada e selecionada.
      - Amarelo: questão em branco.
      - Vermelho: questão ambígua.
      - Cinza:   bolha não marcada.
    """
    debug = warped_bgr.copy()


    x_positions = config["x_positions"]
    y_positions = config["y_positions"]
    bubble_radius = config["bubble_radius"]
    inner_mask_radius = config["inner_mask_radius"]
    grid_2d = config.get("grid", {})
    has_2d_grid = "grid" in config

    for q_str, default_cy in y_positions.items():
        q_num = int(q_str)
        
        # Filtra os resultados dessa questão
        qr = next((qr for qr in question_results if qr.question == q_num), None)
        bubbles = [b for b in bubble_results if b.question == q_num]
        
        for alt in ["A", "B", "C", "D", "E"]:
            if has_2d_grid and q_str in grid_2d and alt in grid_2d[q_str]:
                ellipse = grid_2d[q_str][alt]
                cx, cy = ellipse[0]
            else:
                cx = x_positions[alt]
                cy = default_cy
                ellipse = ((cx, cy), (bubble_radius*2, bubble_radius*2), 0.0)
                
            # Acha o resultado da bolha
            br = next((b for b in bubbles if b.alternative == alt), None)
            if not br or not qr:
                continue

            # Determina a cor do círculo interno
            if qr.is_blank:
                color = COLOR_BLANK
            elif qr.is_ambiguous:
                color = COLOR_AMBIGUOUS
            elif qr.answer == br.alternative:
                color = COLOR_MARKED
            else:
                color = COLOR_UNMARKED

            (cx, cy), (a, b), angle = ellipse
            center = (int(cx), int(cy))
            axes_outer = (int(a/2), int(b/2))
            
            inner_ratio = inner_mask_radius / max(1, bubble_radius)
            axes_inner = (int(a/2 * inner_ratio), int(b/2 * inner_ratio))

            # Círculo externo (borda da bolha real ou deformada)
            cv2.ellipse(debug, center, axes_outer, angle, 0, 360, (100, 100, 100), 1)
            # Círculo interno (região analisada) com preenchimento semitransparente
            overlay = debug.copy()
            cv2.ellipse(overlay, center, axes_inner, angle, 0, 360, color, -1)
            cv2.addWeighted(overlay, 0.35, debug, 0.65, 0, debug)
            # Borda do círculo interno
            cv2.ellipse(debug, center, axes_inner, angle, 0, 360, color, 2)

            # Exibe fill_ratio
            pct_text = f"{br.fill_ratio:.2f}"
            _put_text_centered(debug, pct_text, int(cx), int(cy),
                               COLOR_TEXT, font_scale=0.38, thickness=1)

    # ── Rótulo de resultado por questão (coluna à direita) ──
    x_label = 800

    for qr in question_results:
        # Calcula y médio da questão
        q_bubbles = [br for br in bubble_results if br.question == qr.question]
        if not q_bubbles:
            continue
        cy_q = q_bubbles[0].cy

        if qr.is_blank:
            color = COLOR_BLANK
            label = "BRANCO"
        elif qr.is_ambiguous:
            color = COLOR_AMBIGUOUS
            label = "AMBIG"
        else:
            color = COLOR_MARKED
            label = qr.answer

        cv2.rectangle(debug, (x_label - 5, cy_q - 18), (x_label + 90, cy_q + 8),
                      color, -1)
        cv2.putText(debug, f"Q{qr.question:02d}:{label}",
                    (x_label, cy_q),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_LIGHT, 1, cv2.LINE_AA)

    # ── Legenda ──
    _draw_legend(debug)

    return debug


def _draw_legend(img: np.ndarray) -> None:
    """Adiciona legenda de cores no canto inferior esquerdo."""
    items = [
        (COLOR_MARKED,    "Marcada"),
        (COLOR_BLANK,     "Em branco"),
        (COLOR_AMBIGUOUS, "Ambigua"),
        (COLOR_UNMARKED,  "Nao marcada"),
    ]
    x0, y0 = 20, img.shape[0] - 20 - len(items) * 28
    for color, text in items:
        cv2.rectangle(img, (x0, y0), (x0 + 20, y0 + 20), color, -1)
        cv2.putText(img, text, (x0 + 28, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y0 += 28


# ──────────────────────────────────────────────
# Funções de salvamento das imagens de debug
# ──────────────────────────────────────────────

def save_debug_images(debug_dir: str,
                      original: np.ndarray,
                      gray: np.ndarray,
                      clahe: np.ndarray,
                      edges: np.ndarray,
                      contour_img: np.ndarray,
                      warped: np.ndarray,
                      warped_binary: np.ndarray,
                      bubbles_debug: np.ndarray,
                      prefix: str = "") -> None:
    """
    Salva as 8 imagens de debug numeradas em debug_dir.
    """
    import os
    os.makedirs(debug_dir, exist_ok=True)

    pfx = f"{prefix}_" if prefix else ""
    pairs = [
        (f"{pfx}01_original.jpg",            original),
        (f"{pfx}02_gray.jpg",                gray),
        (f"{pfx}03_clahe.jpg",               clahe),
        (f"{pfx}04_edges_or_threshold.jpg",  edges),
        (f"{pfx}05_contour_detected.jpg",    contour_img),
        (f"{pfx}06_warped.jpg",              warped),
        (f"{pfx}07_threshold_warped.jpg",    warped_binary),
        (f"{pfx}08_bubbles_debug.jpg",       bubbles_debug),
    ]

    for filename, img in pairs:
        path = os.path.join(debug_dir, filename)
        if img is not None:
            cv2.imwrite(path, img)
            print(f"  [debug] Salvo: {path}")
        else:
            print(f"  [debug] Imagem ausente, pulando: {filename}")
