"""
output.py — Formatação e salvamento dos resultados.

Gera:
  - Saída formatada no terminal.
  - data/output/respostas_lidas.json
  - data/output/respostas_lidas.csv
"""

import csv
import json
import os
from reader.bubbles import QuestionResult, ALTERNATIVES


# ──────────────────────────────────────────────
# Saída no terminal
# ──────────────────────────────────────────────

def print_results(question_results: list[QuestionResult]) -> None:
    """Exibe a tabela de resultados formatada no terminal."""
    line = "-" * 60
    print("\n" + line)
    print("  LEITURA DO CARTAO-RESPOSTA")
    print(line)
    header = f"{'Questao':^8} | {'Resposta lida':^15} | {'Confianca / Preenchimento'}"
    print(header)
    print(line)

    for qr in question_results:
        q_str = f"{qr.question:02d}"

        if qr.is_ambiguous:
            ambig_detail = ", ".join(
                f"{alt}={v:.2f}"
                for alt, v in sorted(qr.fill_ratios.items(), key=lambda kv: kv[1], reverse=True)
                if v >= 0.05
            )
            confidence_str = ambig_detail
        else:
            confidence_str = f"{qr.confidence:.2f}"

        print(f"  {q_str:^6}  | {qr.answer:^15} | {confidence_str}")

    print(line + "\n")


# ──────────────────────────────────────────────
# Serialização para JSON
# ──────────────────────────────────────────────

def _question_to_dict(qr: QuestionResult) -> dict:
    return {
        "questao": qr.question,
        "resposta": qr.answer,
        "confianca": round(qr.confidence, 4),
        "em_branco": qr.is_blank,
        "ambigua": qr.is_ambiguous,
        "preenchimento": {alt: round(v, 4) for alt, v in qr.fill_ratios.items()},
    }


def save_json(question_results: list[QuestionResult], output_dir: str, prefix: str = "") -> str:
    """Salva os resultados em formato JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{prefix}_respostas_lidas.json" if prefix else "respostas_lidas.json"
    path = os.path.join(output_dir, filename)

    data = {
        "total_questoes": len(question_results),
        "questoes": [_question_to_dict(qr) for qr in question_results],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  [output] JSON salvo: {path}")
    return path


def save_csv(question_results: list[QuestionResult], output_dir: str, prefix: str = "") -> str:
    """Salva os resultados em formato CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{prefix}_respostas_lidas.csv" if prefix else "respostas_lidas.csv"
    path = os.path.join(output_dir, filename)

    fieldnames = ["questao", "resposta", "confianca", "em_branco", "ambigua",
                  "fill_A", "fill_B", "fill_C", "fill_D", "fill_E"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qr in question_results:
            row = {
                "questao": f"{qr.question:02d}",
                "resposta": qr.answer,
                "confianca": f"{qr.confidence:.4f}",
                "em_branco": qr.is_blank,
                "ambigua": qr.is_ambiguous,
            }
            for alt in ALTERNATIVES:
                row[f"fill_{alt}"] = f"{qr.fill_ratios.get(alt, 0.0):.4f}"
            writer.writerow(row)

    print(f"  [output] CSV salvo: {path}")
    return path


def save_all_outputs(question_results: list[QuestionResult], output_dir: str, prefix: str = "") -> dict[str, str]:
    """Salva JSON e CSV e retorna dict com os caminhos."""
    return {
        "json": save_json(question_results, output_dir, prefix),
        "csv": save_csv(question_results, output_dir, prefix),
    }
