"""
Detecta grupos de tweets casi-duplicados (spam/bots/plantillas virales) usando
MinHash + LSH sobre shingles de palabras, y asigna un grupo de duplicado por
tweet.

Por qué esto importa para el estudio: tweets casi-duplicados (ej. un tweet-
plantilla repetido con una frase inicial distinta cada vez, común en spam o
contenido tipo bot) generan clusters artificialmente "limpios" en el embedding
space que NO reflejan sentido real -- ver caso "aguirre": el mismo
tweet-plantilla repetido se detectó como si tuviera múltiples sentidos, cuando
en realidad es un apellido monosémico. Corre esto ANTES de la extracción de
embeddings, y usa `es_representante=True` para quedarte con un solo tweet por
grupo de casi-duplicados.

Requiere: datasketch, pandas
Ejecuta: pip install datasketch pandas --break-system-packages

Uso:
    python detectar_duplicados.py
"""

import re
from collections import defaultdict

import pandas as pd
from datasketch import MinHash, MinHashLSH

TEXTOS_CSV = "output/all_tweets.csv"  # columnas: id_tweet, texto
SALIDA_CSV = "output/all_tweets_grupos_duplicados.csv"
N_SHINGLE = 3          # tamaño del n-grama de palabras
UMBRAL_JACCARD = 0.75  # similitud mínima para considerar casi-duplicado
NUM_PERM = 128         # permutaciones del MinHash (más = más preciso, más lento)


def normalizar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\sáéíóúñü]", " ", texto)  # quita puntuación, conserva letras/acentos
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def shingles(texto, n=N_SHINGLE):
    palabras = texto.split()
    if len(palabras) < n:
        return {texto} if texto else set()
    return {" ".join(palabras[i : i + n]) for i in range(len(palabras) - n + 1)}


def construir_minhash(shingle_set, num_perm=NUM_PERM):
    m = MinHash(num_perm=num_perm)
    for s in shingle_set:
        m.update(s.encode("utf8"))
    return m


def main():
    df = pd.read_csv(TEXTOS_CSV)
    df["texto_norm"] = df["texto"].map(normalizar)

    lsh = MinHashLSH(threshold=UMBRAL_JACCARD, num_perm=NUM_PERM)
    minhashes = {}

    print("Construyendo MinHash por tweet...")
    for _, row in df.iterrows():
        sh = shingles(row["texto_norm"])
        if not sh:
            continue
        mh = construir_minhash(sh)
        minhashes[row["id_tweet"]] = mh
        lsh.insert(row["id_tweet"], mh)

    # Union-Find para agrupar componentes conexas de casi-duplicados
    padre = {tid: tid for tid in minhashes}

    def encontrar(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def unir(x, y):
        rx, ry = encontrar(x), encontrar(y)
        if rx != ry:
            padre[rx] = ry

    print("Buscando vecinos casi-duplicados por LSH...")
    for i, tid in enumerate(minhashes):
        vecinos = lsh.query(minhashes[tid])
        for v in vecinos:
            if v != tid:
                unir(tid, v)
        if i % 5000 == 0:
            print(f"  {i}/{len(minhashes)} tweets procesados")

    grupos = defaultdict(list)
    for tid in minhashes:
        grupos[encontrar(tid)].append(tid)

    id_a_grupo = {}
    tamano_grupo = {}
    for i, (raiz, miembros) in enumerate(grupos.items()):
        for tid in miembros:
            id_a_grupo[tid] = i
            tamano_grupo[tid] = len(miembros)

    df["grupo_duplicado"] = df["id_tweet"].map(id_a_grupo)
    df["tamano_grupo_duplicado"] = df["id_tweet"].map(tamano_grupo)
    df["es_representante"] = ~df.duplicated(subset="grupo_duplicado", keep="first")

    n_grupos = df["grupo_duplicado"].nunique()
    n_en_grupo = (df["tamano_grupo_duplicado"] > 1).sum()
    print(f"\nTotal tweets: {len(df)}")
    print(f"Grupos de casi-duplicados encontrados: {n_grupos}")
    print(f"Tweets que pertenecen a un grupo con >1 miembro: {n_en_grupo} ({n_en_grupo/len(df)*100:.1f}%)")

    df[["id_tweet", "grupo_duplicado", "tamano_grupo_duplicado", "es_representante"]].to_csv(SALIDA_CSV, index=False)
    print(f"\nGuardado: {SALIDA_CSV}")

    print("\nGrupos más grandes (top 10, tamaño):")
    top = df.groupby("grupo_duplicado").size().sort_values(ascending=False).head(10)
    print(top)


if __name__ == "__main__":
    main()