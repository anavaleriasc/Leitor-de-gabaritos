"""
main.py — Ponto de entrada do sistema de leitura de cartão-resposta.

Uso:
    python main.py --image data/input/cartao01.jpg
    python main.py --image data/input/cartao01.jpg --gabarito "1-A,2-C,3-D,4-B,5-E,6-A,7-C,8-B,9-D,10-E"
    python main.py --image data/input/cartao01.jpg --config config/sheet_config.json
    python main.py --image data/input/cartao01.jpg --no-warp  # pula correção de perspectiva

Flags:
    --image     Caminho para a foto do cartão-resposta (obrigatório).
    --gabarito  String com gabarito oficial (opcional).
    --config    Caminho alternativo para sheet_config.json (padrão: config/sheet_config.json).
    --no-warp   Pula a etapa de correção de perspectiva (útil para testes com imagem já reta).
    --debug-dir Diretório para imagens de debug (padrão: data/debug).
    --output-dir Diretório para arquivos de saída (padrão: data/output).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

from reader.preprocessing import (
    to_grayscale,
    apply_clahe,
    preprocess_for_contour_detection,
    preprocess_warped_for_bubbles,
)
from reader.perspective import (
    detect_and_warp,
    draw_detected_contour,
    warp_perspective,
)
from reader.calibration import auto_calibrate_grid
from reader.bubbles import analyze_all_bubbles
from reader.output import print_results, save_all_outputs
from reader.visualization import draw_bubbles_debug, save_debug_images
from reader.grading import parse_answer_key, grade, print_grading_report


# ──────────────────────────────────────────────
# Configuração padrão (usada se o JSON não existir)
# ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    "warped_width": 1000,
    "warped_height": 1400,
    "bubble_radius": 22,
    "inner_mask_radius": 14,
    "min_fill_ratio": 0.18,
    "ambiguity_margin": 0.08,
    "x_positions": {"A": 300, "B": 410, "C": 520, "D": 630, "E": 740},
    "y_positions": {
        "1": 330, "2": 420, "3": 510, "4": 600, "5": 690,
        "6": 780, "7": 870, "8": 960, "9": 1050, "10": 1140,
    },
}


# ──────────────────────────────────────────────
# Utilitários
# ──────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Carrega configuração do JSON; usa defaults se o arquivo não existir."""
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"[config] Configuração carregada de: {config_path}")
        return cfg
    else:
        print(f"[config] Arquivo '{config_path}' não encontrado. Usando valores padrão.")
        return DEFAULT_CONFIG


def load_image(path: str) -> np.ndarray:
    """Carrega a imagem do caminho fornecido com tratamento de erro."""
    if not os.path.isfile(path):
        print(f"[ERRO] Arquivo não encontrado: {path}", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(path)
    if image is None:
        print(f"[ERRO] Não foi possível carregar a imagem: {path}", file=sys.stderr)
        print("       Verifique se o arquivo é uma imagem válida (JPG, PNG, etc.).",
              file=sys.stderr)
        sys.exit(1)

    print(f"[info] Imagem carregada: {path}  ({image.shape[1]}×{image.shape[0]} px)")
    return image


def ensure_dirs(*dirs: str) -> None:
    """Cria diretórios necessários se não existirem."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────

def run_pipeline(image_path: str,
                 config: dict,
                 debug_dir: str,
                 output_dir: str,
                 skip_warp: bool,
                 auto_calibrate: bool,
                 use_markers: bool) -> list:
    """
    Executa o pipeline completo de PDI e retorna a lista de QuestionResult.

    Etapas:
      1. Carregar imagem.
      2. Converter para tons de cinza + CLAHE.
      3. Detectar bordas (Canny).
      4. Encontrar contorno da folha e corrigir perspectiva.
      5. Pré-processar imagem corrigida (limiarização adaptativa).
      6. Analisar bolhas por coordenadas fixas.
      7. Salvar saídas e imagens de debug.
    """
    if use_markers:
        config["bubble_radius"] = 45
        config["inner_mask_radius"] = 33
    else:
        # Valores originais matematicamente ajustados para a imagem A4
        config["bubble_radius"] = 22
        config["inner_mask_radius"] = 15
        
    out_w = config["warped_width"]
    out_h = config["warped_height"]

    # ── Etapa 1: Carregar ──────────────────────────────────────────
    original = load_image(image_path)
    import os
    file_prefix = os.path.splitext(os.path.basename(image_path))[0]

    # ── Etapa 2: Tons de cinza + CLAHE ────────────────────────────
    print("\n[pipeline] Etapa 2: Conversão para tons de cinza e CLAHE...")
    gray, clahe_img, edges = preprocess_for_contour_detection(original, use_markers=use_markers)

    # ── Etapa 3: Correção de perspectiva ──────────────────────────
    contour_debug_img = original.copy()
    rect = None

    if skip_warp:
        print("[pipeline] Etapa 3: Correção de perspectiva PULADA (--no-warp).")
        # Usa a imagem original redimensionada como "warped"
        warped_bgr = cv2.resize(original, (out_w, out_h))
        warped_gray = cv2.resize(gray, (out_w, out_h))
    else:
        print("[pipeline] Etapa 3: Detectando contorno da folha (marcadores=" + str(use_markers) + ")...")
        warped_bgr, rect = detect_and_warp(original, edges, out_w, out_h, use_markers=use_markers)

        if warped_bgr is None and not use_markers:
            # ── Fallback: tenta com limiarização em vez de Canny ──
            print("[pipeline]   Canny não encontrou contorno. Tentando limiarização...")
            _, bin_fallback = cv2.threshold(
                clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            import cv2 as _cv2
            kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (5, 5))
            bin_fallback = _cv2.dilate(bin_fallback, kernel, iterations=2)

            from reader.perspective import detect_and_warp as _daw
            warped_bgr, rect = _daw(original, bin_fallback, out_w, out_h, use_markers=False)

        if warped_bgr is None:
            print("\n[ERRO] Não foi possível detectar o contorno da folha.", file=sys.stderr)
            print("       Dicas para melhorar a detecção:", file=sys.stderr)
            print("         • Use um fundo escuro (ex.: mesa escura ou cartolina preta).", file=sys.stderr)
            print("         • Garanta boa iluminação uniforme sem reflexos.", file=sys.stderr)
            print("         • Enquadre a folha sem cortar bordas.", file=sys.stderr)
            print("         • Use --no-warp se a imagem já estiver alinhada.", file=sys.stderr)
            # Salva imagem de debug com bordas detectadas para diagnóstico
            ensure_dirs(debug_dir)
            pfx = f"{file_prefix}_" if file_prefix else ""
            cv2.imwrite(os.path.join(debug_dir, f"{pfx}01_original.jpg"), original)
            cv2.imwrite(os.path.join(debug_dir, f"{pfx}02_gray.jpg"), gray)
            cv2.imwrite(os.path.join(debug_dir, f"{pfx}03_clahe.jpg"), clahe_img)
            cv2.imwrite(os.path.join(debug_dir, f"{pfx}04_edges_or_threshold.jpg"), edges)
            
            # Para fins de relatório didático: desenhar todos os contornos soltos que o OpenCV encontrou
            # Isso ajuda a provar visualmente por que o fundo claro causou falha
            crash_contours_img = original.copy()
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crash_contours_img, cnts, -1, (0, 0, 255), 2)  # Vermelho
            cv2.imwrite(os.path.join(debug_dir, f"{pfx}05_contour_detected.jpg"), crash_contours_img)
            
            print(f"\n  [debug] Imagens parciais salvas em: {debug_dir}")
            sys.exit(1)

        warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

        if rect is not None:
            contour_debug_img = draw_detected_contour(original, rect)
            print(f"[pipeline]   Contorno detectado com sucesso.")

    # ── Etapa 4: Limiarização da imagem corrigida ──────────────────
    print("[pipeline] Etapa 4: Limiarização adaptativa da imagem corrigida...")
    warped_binary = preprocess_warped_for_bubbles(warped_gray)

    # ── Etapa 4.5: Calibração Automática ──────────────────────────
    if auto_calibrate:
        print("[pipeline] Etapa 4.5: Calibrando coordenadas automaticamente...")
        calib_result = auto_calibrate_grid(warped_gray, config["bubble_radius"])
        if calib_result:
            x_pos, y_pos, grid_2d = calib_result
            config["x_positions"] = x_pos
            config["y_positions"] = y_pos
            config["grid"] = grid_2d
            print("[pipeline]   Calibração bem-sucedida! Coordenadas e homografia atualizadas.")
        else:
            print("[AVISO] Calibração automática falhou. Usando coordenadas do sheet_config.json.")

    # ── Etapa 5: Análise das bolhas ────────────────────────────────
    print("[pipeline] Etapa 5: Analisando bolhas...")
    bubble_results, question_results = analyze_all_bubbles(warped_binary, config)

    # ── Etapa 6: Gerar imagem de debug das bolhas ──────────────────
    print("[pipeline] Etapa 6: Gerando visualização de debug das bolhas...")
    bubbles_debug_img = draw_bubbles_debug(
        warped_bgr,
        bubble_results,
        question_results,
        config
    )


    # ── Etapa 7: Salvando imagens de debug ──────────────────────────
    print("[pipeline] Etapa 7: Salvando imagens de debug...")
    ensure_dirs(debug_dir, output_dir)

    # Converte imagens de um canal para BGR para salvar com cv2
    def to_bgr_debug(img):
        if img is None:
            return None
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    save_debug_images(
        debug_dir=debug_dir,
        original=original,
        gray=to_bgr_debug(gray),
        clahe=to_bgr_debug(clahe_img),
        edges=to_bgr_debug(edges),
        contour_img=contour_debug_img,
        warped=warped_bgr,
        warped_binary=to_bgr_debug(warped_binary),
        bubbles_debug=bubbles_debug_img,
        prefix=file_prefix,
    )

    # ── Etapa 8: Salvar resultados ────────────────────────────────
    print("[pipeline] Etapa 8: Salvando resultados...")
    save_all_outputs(question_results, output_dir, prefix=file_prefix)

    return question_results


# ──────────────────────────────────────────────
# Ponto de entrada
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sistema de leitura automática de cartão-resposta via PDI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py --image data/input/cartao01.jpg
  python main.py --image data/input/cartao01.jpg --gabarito "1-A,2-C,3-D,4-B,5-E,6-A,7-C,8-B,9-D,10-E"
  python main.py --image data/input/cartao01.jpg --no-warp
        """,
    )
    parser.add_argument(
        "--image", required=True,
        help="Caminho para a imagem/foto do cartão-resposta.",
    )
    parser.add_argument(
        "--gabarito", default=None,
        help='Gabarito oficial (opcional). Ex.: "1-A,2-C,3-D,4-B,5-E,6-A,7-C,8-B,9-D,10-E"',
    )
    parser.add_argument(
        "--config", default="config/sheet_config.json",
        help="Caminho para o arquivo de configuração JSON (padrão: config/sheet_config.json).",
    )
    parser.add_argument(
        "--no-warp", action="store_true",
        help="Pula a correção de perspectiva. Use se a imagem já estiver alinhada.",
    )
    parser.add_argument(
        "--auto-calibrate", action="store_true",
        help="Detecta automaticamente as coordenadas das bolhas ignorando o JSON.",
    )
    parser.add_argument(
        "--debug-dir", default="data/debug",
        help="Diretório para imagens de debug (padrão: data/debug).",
    )
    parser.add_argument(
        "--output-dir", default="data/output",
        help="Diretório para arquivos de saída (padrão: data/output).",
    )
    parser.add_argument(
        "--use-markers", action="store_true",
        help="Habilita a detecção de 4 fiduciais (retângulos pretos) nos cantos em vez de usar as bordas da folha.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  SISTEMA DE LEITURA DE CARTAO-RESPOSTA")
    print("  Tecnicas Classicas de Processamento Digital de Imagens")
    print("=" * 60)

    # Carrega configuração
    config = load_config(args.config)

    # Executa pipeline
    question_results = run_pipeline(
        image_path=args.image,
        config=config,
        debug_dir=args.debug_dir,
        output_dir=args.output_dir,
        skip_warp=args.no_warp,
        auto_calibrate=args.auto_calibrate,
        use_markers=args.use_markers,
    )

    # Exibe resultados no terminal
    print_results(question_results)

    # ── Módulo opcional: comparação com gabarito ──
    if args.gabarito:
        try:
            answer_key = parse_answer_key(args.gabarito)
            grading_results = grade(question_results, answer_key)
            print_grading_report(grading_results)
        except ValueError as e:
            print(f"\n[AVISO] Gabarito inválido: {e}", file=sys.stderr)

    print(f"[info] Imagens de debug salvas em : {args.debug_dir}/")
    print(f"[info] Resultados salvos em       : {args.output_dir}/")
    print("[info] Concluído.\n")


if __name__ == "__main__":
    main()
