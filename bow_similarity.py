"""
Practica: Similitud de documentos con Bag-of-Words y similitud del coseno (PyTorch).

Pipeline:
  1. Leer los documentos .docx.
  2. Preprocesar el texto (minusculas, quitar puntuacion/no alfabeticos, stopwords).
  3. Construir un vocabulario global y representar cada documento como un
     vector de frecuencias (bag-of-words) usando tensores de PyTorch.
  4. Calcular la matriz de similitud del coseno entre todos los pares de
     documentos con torch.nn.functional.cosine_similarity.
  5. Analizar los pares mas/menos similares.
"""

import glob
import re
import unicodedata
from pathlib import Path

import docx
import torch
import torch.nn.functional as F

DOCS_DIR = Path(__file__).parent

# Stopwords en espanol (lista compacta, sin dependencias externas tipo nltk)
STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante", "e",
    "el", "ella", "ellas", "ellos", "en", "entre", "era", "erais", "eran",
    "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta",
    "estaba", "estabais", "estaban", "estabas", "estad", "estada", "estadas",
    "estado", "estados", "estamos", "estan", "estando", "estar", "estara",
    "estaran", "estaras", "estare", "estareis", "estaremos", "estaria",
    "estariais", "estariamos", "estarian", "estarias", "estas", "este",
    "estemos", "esto", "estos", "estoy", "etc", "fue", "fuera", "fuerais",
    "fueran", "fueras", "fueron", "fuese", "fueseis", "fuesen", "fueses",
    "fui", "fuimos", "ha", "habeis", "haber", "habia", "habiais", "habiamos",
    "habian", "habias", "habida", "habidas", "habido", "habidos", "habiendo",
    "han", "has", "hasta", "hay", "haya", "hayamos", "hayan", "hayas",
    "hayais", "he", "hemos", "hube", "hubiera", "hubierais", "hubieran",
    "hubieras", "hubieron", "hubiese", "hubieseis", "hubiesen", "hubieses",
    "hubimos", "hubiste", "hubisteis", "hubo", "la", "las", "le", "les",
    "lo", "los", "mas", "me", "mi", "mientras", "mio", "mis", "misma",
    "mismas", "mismo", "mismos", "mucho", "muchos", "muy", "nada", "ni",
    "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro",
    "nuestros", "o", "os", "otra", "otras", "otro", "otros", "para",
    "pero", "poco", "por", "porque", "que", "quien", "quienes", "se",
    "sea", "seamos", "sean", "seas", "sera", "seran", "seras", "sere",
    "sereis", "seremos", "seria", "seriais", "seriamos", "serian",
    "serias", "si", "sido", "siendo", "sin", "sobre", "sois", "somos",
    "son", "soy", "su", "sus", "suya", "suyas", "suyo", "suyos", "tal",
    "tambien", "tanto", "te", "tendra", "tendran", "tendras", "tendre",
    "tendreis", "tendremos", "tendria", "tendriais", "tendriamos",
    "tendrian", "tendrias", "tened", "tenemos", "tenga", "tengamos",
    "tengan", "tengas", "tengo", "tenia", "teniais", "teniamos", "tenian",
    "tenias", "ti", "tiene", "tienen", "tienes", "todo", "todos", "tu",
    "tus", "tuya", "tuyas", "tuyo", "tuyos", "un", "una", "uno", "unos",
    "vosotras", "vosotros", "vuestra", "vuestras", "vuestro", "vuestros",
    "y", "ya", "yo",
}


def read_docx(path: Path) -> str:
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = strip_accents(text)
    text = re.sub(r"[^a-z\s]", " ", text)  # solo caracteres alfabeticos
    tokens = text.split()
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]


def build_vocabulary(tokenized_docs: list[list[str]]) -> list[str]:
    vocab = sorted({tok for doc in tokenized_docs for tok in doc})
    return vocab


def build_bow_matrix(tokenized_docs: list[list[str]], vocab: list[str]) -> torch.Tensor:
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    matrix = torch.zeros((len(tokenized_docs), len(vocab)), dtype=torch.float32)
    for i, doc in enumerate(tokenized_docs):
        for tok in doc:
            matrix[i, word_to_idx[tok]] += 1.0
    return matrix


def cosine_similarity_matrix(bow: torch.Tensor) -> torch.Tensor:
    """Similitud del coseno entre todas las filas de `bow` usando PyTorch."""
    n = bow.size(0)
    a = bow.unsqueeze(1).expand(n, n, bow.size(1))
    b = bow.unsqueeze(0).expand(n, n, bow.size(1))
    return F.cosine_similarity(a, b, dim=2)


def main() -> None:
    paths = sorted(DOCS_DIR.glob("doc*.docx"))
    names = [p.stem for p in paths]
    raw_texts = [read_docx(p) for p in paths]

    tokenized_docs = [preprocess(t) for t in raw_texts]
    vocab = build_vocabulary(tokenized_docs)
    bow = build_bow_matrix(tokenized_docs, vocab)

    print(f"Documentos procesados: {len(names)}")
    print(f"Tamano del vocabulario: {len(vocab)}")
    print()

    sim = cosine_similarity_matrix(bow)

    # Matriz de similitud
    header = "        " + " ".join(f"{n:>8}" for n in names)
    print("Matriz de similitud del coseno:")
    print(header)
    for i, name in enumerate(names):
        row = " ".join(f"{sim[i, j].item():8.3f}" for j in range(len(names)))
        print(f"{name:>8} {row}")
    print()

    # Pares mas y menos similares (excluyendo diagonal)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((sim[i, j].item(), names[i], names[j]))
    pairs.sort(reverse=True)

    print("Top 5 pares MAS similares:")
    for score, a, b in pairs[:5]:
        print(f"  {a} - {b}: {score:.3f}")

    print()
    print("Top 5 pares MENOS similares:")
    for score, a, b in pairs[-5:]:
        print(f"  {a} - {b}: {score:.3f}")


if __name__ == "__main__":
    main()
