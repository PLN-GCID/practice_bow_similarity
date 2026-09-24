"""
Practica: Similitud de documentos con TF-IDF y similitud del coseno (PyTorch).

Pipeline:
  1. Reutilizar la lectura, el preprocesamiento y la matriz Bag-of-Words de
     la parte 1 (bow/bow_similarity.py), para que ambos modelos partan del
     mismo vocabulario.
  2. Ponderar la matriz de frecuencias con TF-IDF usando tensores de PyTorch.
  3. Calcular la matriz de similitud del coseno entre todos los pares de
     documentos.
  4. Comparar los resultados con los del modelo BoW.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "bow"))

from bow_similarity import (  # noqa: E402
    DOCS_DIR,
    build_bow_matrix,
    build_vocabulary,
    cosine_similarity_matrix,
    preprocess,
    read_docx,
)


# Tema de cada documento (ver bow/informe_bow.md)
TOPICS = {
    "doc01": "movil", "doc02": "movil", "doc03": "movil", "doc06": "movil",
    "doc07": "movil", "doc10": "movil",
    "doc04": "ve", "doc05": "ve", "doc08": "ve", "doc09": "ve",
    "doc11": "hibrido", "doc12": "hibrido",
}

GROUP_PAIRS = [
    ("movil", "movil"),
    ("ve", "ve"),
    ("movil", "ve"),
    ("hibrido", "movil"),
    ("hibrido", "ve"),
]


def compute_idf(bow: torch.Tensor) -> torch.Tensor:
    """IDF suavizado: log((1 + N) / (1 + df)) + 1.

    El suavizado evita divisiones por cero y que una palabra presente en
    todos los documentos reciba peso nulo.
    """
    n_docs = bow.size(0)
    df = (bow > 0).sum(dim=0).to(torch.float32)
    return torch.log((1.0 + n_docs) / (1.0 + df)) + 1.0


def build_tfidf_matrix(bow: torch.Tensor) -> torch.Tensor:
    """TF (frecuencia relativa en el documento) x IDF."""
    tf = bow / bow.sum(dim=1, keepdim=True)
    return tf * compute_idf(bow)


def sorted_pairs(sim: torch.Tensor, names: list[str]) -> list[tuple[float, str, str]]:
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((sim[i, j].item(), names[i], names[j]))
    pairs.sort(reverse=True)
    return pairs


def group_mean(sim: torch.Tensor, names: list[str], g1: str, g2: str) -> float:
    """Similitud media entre pares de documentos (distintos) de los grupos g1 y g2."""
    values = [
        sim[i, j].item()
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if {TOPICS[names[i]], TOPICS[names[j]]} == {g1, g2}
    ]
    return sum(values) / len(values)


def print_matrix(title: str, matrix: torch.Tensor, names: list[str]) -> None:
    print(title)
    print("        " + " ".join(f"{n:>8}" for n in names))
    for i, name in enumerate(names):
        row = " ".join(f"{matrix[i, j].item():8.3f}" for j in range(len(names)))
        print(f"{name:>8} {row}")
    print()


def main() -> None:
    paths = sorted(DOCS_DIR.glob("doc*.docx"))
    names = [p.stem for p in paths]
    raw_texts = [read_docx(p) for p in paths]

    tokenized_docs = [preprocess(t) for t in raw_texts]
    vocab = build_vocabulary(tokenized_docs)
    bow = build_bow_matrix(tokenized_docs, vocab)
    tfidf = build_tfidf_matrix(bow)

    print(f"Documentos procesados: {len(names)}")
    print(f"Tamano del vocabulario: {len(vocab)}")
    print()

    # Palabras con menor y mayor IDF
    idf = compute_idf(bow)
    order = torch.argsort(idf)
    print("Palabras mas comunes (IDF mas bajo):")
    print("  " + ", ".join(f"{vocab[i]} ({idf[i].item():.2f})" for i in order[:10].tolist()))
    print()

    sim_tfidf = cosine_similarity_matrix(tfidf)
    sim_bow = cosine_similarity_matrix(bow)

    print_matrix("Matriz de similitud del coseno (TF-IDF):", sim_tfidf, names)

    pairs = sorted_pairs(sim_tfidf, names)
    print("Top 5 pares MAS similares (TF-IDF):")
    for score, a, b in pairs[:5]:
        print(f"  {a} - {b}: {score:.3f}")
    print()
    print("Top 5 pares MENOS similares (TF-IDF):")
    for score, a, b in pairs[-5:]:
        print(f"  {a} - {b}: {score:.3f}")
    print()

    # Comparacion con BoW
    print_matrix("Diferencia TF-IDF - BoW:", sim_tfidf - sim_bow, names)

    bow_rank = {(a, b): r for r, (_, a, b) in enumerate(sorted_pairs(sim_bow, names), start=1)}
    print("Top 10 pares TF-IDF comparados con BoW:")
    print(f"  {'par':<14} {'TF-IDF':>7} {'BoW':>7}  rango TF-IDF <- BoW")
    for rank, (score, a, b) in enumerate(pairs[:10], start=1):
        i, j = names.index(a), names.index(b)
        print(f"  {a + ' - ' + b:<14} {score:7.3f} {sim_bow[i, j].item():7.3f} "
              f"  {rank:>6} <- {bow_rank[(a, b)]}")
    print()

    print("Similitud media por grupos tematicos:")
    print(f"  {'grupos':<16} {'BoW':>7} {'TF-IDF':>7}")
    for g1, g2 in GROUP_PAIRS:
        print(f"  {g1 + ' - ' + g2:<16} {group_mean(sim_bow, names, g1, g2):7.3f} "
              f"{group_mean(sim_tfidf, names, g1, g2):7.3f}")
    print()

    mask = ~torch.eye(len(names), dtype=torch.bool)
    print(f"Similitud media fuera de la diagonal: BoW = {sim_bow[mask].mean().item():.3f}, "
          f"TF-IDF = {sim_tfidf[mask].mean().item():.3f}")


if __name__ == "__main__":
    main()
