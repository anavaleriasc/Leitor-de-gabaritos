"""
grading.py — Módulo opcional de comparação com gabarito oficial.

Uso:
    python main.py --image data/input/cartao01.jpg --gabarito "1-A,2-C,3-D,4-B,5-E,6-A,7-C,8-B,9-D,10-E"

Este módulo é completamente independente da leitura do cartão. O sistema
funciona sem gabarito; este módulo só é ativado se --gabarito for fornecido.
"""

from __future__ import annotations
from dataclasses import dataclass
from reader.bubbles import QuestionResult


@dataclass
class GradingResult:
    """Resultado da comparação de uma questão com o gabarito."""
    question: int
    read_answer: str
    correct_answer: str
    is_correct: bool


def parse_answer_key(answer_key_str: str) -> dict[int, str]:
    """
    Faz o parse da string de gabarito no formato "1-A,2-C,3-D,...".

    Returns:
        Dicionário {numero_questão: alternativa_correta}.

    Raises:
        ValueError: se o formato for inválido.
    """
    answer_key: dict[int, str] = {}
    parts = answer_key_str.strip().split(",")

    for part in parts:
        part = part.strip()
        if "-" not in part:
            raise ValueError(
                f"Formato inválido no gabarito: '{part}'. "
                "Use o formato '1-A,2-C,...'"
            )
        q_str, alt = part.split("-", maxsplit=1)
        try:
            q_num = int(q_str.strip())
        except ValueError:
            raise ValueError(f"Número de questão inválido: '{q_str}'")

        alt = alt.strip().upper()
        if alt not in ("A", "B", "C", "D", "E"):
            raise ValueError(
                f"Alternativa inválida '{alt}' para questão {q_num}. "
                "Use A, B, C, D ou E."
            )
        answer_key[q_num] = alt

    return answer_key


def grade(question_results: list[QuestionResult],
          answer_key: dict[int, str]) -> list[GradingResult]:
    """
    Compara as respostas lidas com o gabarito oficial.

    Questões com EM_BRANCO ou AMBIGUA são sempre consideradas erradas.
    """
    grading_results: list[GradingResult] = []

    for qr in question_results:
        correct = answer_key.get(qr.question)
        if correct is None:
            continue

        is_correct = (qr.answer == correct)
        grading_results.append(GradingResult(
            question=qr.question,
            read_answer=qr.answer,
            correct_answer=correct,
            is_correct=is_correct,
        ))

    return grading_results


def print_grading_report(grading_results: list[GradingResult]) -> None:
    """Exibe o relatório de acertos e erros no terminal."""
    total = len(grading_results)
    correct_count = sum(1 for gr in grading_results if gr.is_correct)
    score_pct = (correct_count / total * 100) if total > 0 else 0.0

    line = "-" * 52
    print("\n" + line)
    print("  COMPARACAO COM GABARITO OFICIAL")
    print(line)
    print(f"  {'Questao':^8} | {'Lida':^8} | {'Gabarito':^8} | {'Status':^8}")
    print(line)

    for gr in grading_results:
        status = "CERTO" if gr.is_correct else "ERRADO"
        print(f"  {gr.question:^8}  | {gr.read_answer:^8} | {gr.correct_answer:^8} | {status}")

    print(line)
    print(f"  Resultado: {correct_count}/{total} ({score_pct:.1f}%)")
    print(line + "\n")
