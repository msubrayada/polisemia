
"""
Calcula la intersección de palabras (word_normalized) que superan un umbral
de frecuencia en TODOS los modelos, a partir de los CSV generados por
reconstruir_palabras.py.

Uso:
    python interseccion_vocabulario.py \
        --files mex_state:mex_state_palabras.csv mex_large:mex_large_palabras.csv \
                beto:beto_palabras.csv bertin:bertin_palabras.csv \
                robertuito:robertuito_palabras.csv \
        --threshold 5 \
        --out interseccion_vocabulario.csv
"""

import argparse
import sys
from functools import reduce

import pandas as pd


def load_freq(path, threshold):
    df = pd.read_csv(path)
    freq = df["word_normalized"].value_counts()
    qualifying = set(freq[freq >= threshold].index)
    return freq, qualifying


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="pares nombre:archivo.csv (salida de reconstruir_palabras.py)",
    )
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--out", default="interseccion_vocabulario.csv")
    args = parser.parse_args()

    model_files = {}
    for item in args.files:
        name, path = item.split(":", 1)
        model_files[name] = path

    freqs = {}
    qualifying_sets = {}
    for name, path in model_files.items():
        freq, qualifying = load_freq(path, args.threshold)
        freqs[name] = freq
        qualifying_sets[name] = qualifying
        print(f"{name}: {len(qualifying)} tipos con frecuencia >= {args.threshold}", file=sys.stderr)

    interseccion = reduce(lambda a, b: a & b, qualifying_sets.values())
    print(
        f"\nIntersección final (palabras presentes en los {len(model_files)} "
        f"modelos, freq >= {args.threshold} en cada uno): {len(interseccion)}",
        file=sys.stderr,
    )

    rows = []
    for word in sorted(interseccion):
        row = {"word_normalized": word}
        for name, freq in freqs.items():
            row[f"freq_{name}"] = int(freq.get(word, 0))
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Guardado en: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
