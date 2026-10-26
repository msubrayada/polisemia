"""
revisar_glosas_juez.py
=======================

Validación del juez LLM ampliada a TODAS las palabras evaluadas en
medir_separabilidad_llm.py (antes solo 9 de 39 tenían etiqueta de la revisión
manual previa).

Dos pasos:

1) generar_revision()
   Para cada palabra evaluable escribe en un .txt los sentidos que propuso el
   juez (glosa, cuántos tweets tiene cada uno) y 3 tweets de ejemplo por
   sentido, más una plantilla CSV para que etiquetes cada palabra.
   La revisión es CIEGA: no muestra ningún resultado de los modelos (ni
   silhouette ni k-NN), para que la etiqueta no se contamine sabiendo qué
   modelo "separó" qué.

   Etiquetas sugeridas para la columna `etiqueta` del CSV:
     sentido -> los grupos del juez corresponden a sentidos léxicos distintos
     tema    -> un mismo sentido partido por tema/dominio/entidad nombrada
     mixto   -> al menos un grupo es un sentido real y otro es partición temática
     ruido   -> spam, copypasta o grupos sin sentido claro

2) analizar_etiquetas()
   Une tus etiquetas con separabilidad_llm_vs_embeddings.csv y compara, por
   modelo, la mejora de k-NN sobre la clase mayoritaria entre palabras
   "sentido" y "tema" (Mann-Whitney, una cola: sentido > tema).

Uso:
    python revisar_glosas_juez.py generar      # crea revision_glosas.txt y etiquetas_glosas.csv
    # ... llenas la columna `etiqueta` en etiquetas_glosas.csv ...
    python revisar_glosas_juez.py analizar
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import json


CARPETA_JUICIOS = "juicios_llm_qwen_model_"
ETIQUETA_RUIDO_LLM = "ruido_llm"
MIN_TWEETS_POR_SENTIDO_LLM = 5


def _asignaciones_con_ruido_llm(juicio, min_tweets=MIN_TWEETS_POR_SENTIDO_LLM):
    """Devuelve (asignaciones_llm, sentidos_ruido, n_sentidos_5plus).
 
    asignaciones_llm: dict id_tweet -> sentido, con los sentidos que tienen
    menos de `min_tweets` tweets asignados (contando TODA la muestra que vio
    el juez, no solo la que se cruza con el modelo) re-etiquetados como
    ETIQUETA_RUIDO_LLM.
    sentidos_ruido: el conjunto de ids de sentido tratados como ruido.
    n_sentidos_5plus: cuántos sentidos declarados por el LLM tienen >=
    min_tweets tweets -- el conteo "real" de sentidos una vez descontado el
    ruido, más comparable con el criterio que ya se usa para decidir si un
    modelo de clustering "presenta múltiples clusters" (sección 8/9).
    """
    conteo = {}
    for a in juicio["asignaciones"]:
        conteo[a["sentido"]] = conteo.get(a["sentido"], 0) + 1
 
    sentidos_ruido = {
        sid for sid, n in conteo.items()
        if sid != "no_claro" and n < min_tweets
    }
    n_sentidos_5plus = sum(
        1 for sid in (s["id"] for s in juicio.get("sentidos", []))
        if conteo.get(sid, 0) >= min_tweets
    )
 
    asignaciones_llm = {
        a["id_tweet"]: (ETIQUETA_RUIDO_LLM if a["sentido"] in sentidos_ruido else a["sentido"])
        for a in juicio["asignaciones"]
    }
    return asignaciones_llm, sentidos_ruido, n_sentidos_5plus


def cargar_juicio_llm(palabra, carpeta=CARPETA_JUICIOS):
    ruta = os.path.join(carpeta, f"{palabra}.json")
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)

# Ajusta a tu archivo de textos: necesita columnas id_tweet y texto (el mismo
# textos_df que usaste con llm_juez_local.py).
RUTA_TEXTOS = "output/all_tweets.csv"
RUTA_SEPARABILIDAD = "separabilidad_llm_vs_embeddings.csv"
RUTA_REVISION = "revision_glosas.txt"
RUTA_ETIQUETAS = "etiquetas_glosas.csv"

N_EJEMPLOS_POR_SENTIDO = 3
MAX_CARACTERES = 220
SEMILLA = 42


def _palabras_evaluables(ruta_separabilidad=RUTA_SEPARABILIDAD):
    """Palabras con resultado válido (no NaN) en al menos un modelo -- las
    mismas que entraron al análisis de separabilidad."""
    df = pd.read_csv(ruta_separabilidad)
    return sorted(df.dropna(subset=["mejora_sobre_base"])["palabra"].unique())


def generar_revision(ruta_textos=RUTA_TEXTOS, ruta_separabilidad=RUTA_SEPARABILIDAD,
                       carpeta_juicios=CARPETA_JUICIOS, ruta_revision=RUTA_REVISION,
                       ruta_etiquetas=RUTA_ETIQUETAS,
                       min_tweets=MIN_TWEETS_POR_SENTIDO_LLM,
                       n_ejemplos=N_EJEMPLOS_POR_SENTIDO):
    textos = pd.read_csv(ruta_textos)
    id_a_texto = dict(zip(textos["id_tweet"], textos["texto"]))
    rng = np.random.default_rng(SEMILLA)

    palabras = _palabras_evaluables(ruta_separabilidad)
    filas_plantilla = []

    with open(ruta_revision, "w", encoding="utf-8") as f:
        f.write(f"Revisión ciega de las particiones del juez LLM -- {len(palabras)} palabras\n")
        f.write("Etiquetas: sentido / tema / mixto / ruido (ver encabezado del script)\n\n")

        for palabra in palabras:
            juicio = cargar_juicio_llm(palabra, carpeta_juicios)
            asignaciones, sentidos_ruido, n_reales = _asignaciones_con_ruido_llm(juicio, min_tweets=min_tweets)
            glosas = {s["id"]: s.get("glosa", "") for s in juicio.get("sentidos", [])}

            por_sentido = {}
            for id_tweet, sid in asignaciones.items():
                por_sentido.setdefault(sid, []).append(id_tweet)

            f.write("=" * 90 + "\n")
            f.write(f"PALABRA: {palabra}   ({n_reales} sentidos con >= {min_tweets} tweets)\n")
            f.write("=" * 90 + "\n")

            reales = [sid for sid in glosas if sid in por_sentido and sid not in sentidos_ruido]
            for sid in reales:
                ids = por_sentido[sid]
                f.write(f"\n  [{sid}] {glosas[sid]}  -- {len(ids)} tweets\n")
                muestra = rng.choice(ids, size=min(n_ejemplos, len(ids)), replace=False)
                for id_tweet in muestra:
                    texto = str(id_a_texto.get(id_tweet, "(texto no encontrado)")).replace("\n", " ")
                    f.write(f"      - {texto[:MAX_CARACTERES]}\n")

            n_ruido_tweets = len(por_sentido.get(ETIQUETA_RUIDO_LLM, []))
            n_no_claro = len(por_sentido.get("no_claro", []))
            descartados = [f"{sid} ({glosas.get(sid, '')})" for sid in sorted(sentidos_ruido)]
            f.write(f"\n  Descartados como ruido (<{min_tweets} tweets): {len(sentidos_ruido)} sentidos, "
                    f"{n_ruido_tweets} tweets" + (f": {'; '.join(descartados)}" if descartados else "") + "\n")
            f.write(f"  no_claro: {n_no_claro} tweets\n\n")

            filas_plantilla.append({
                "palabra": palabra,
                "n_sentidos_reales": n_reales,
                "glosas": " | ".join(glosas[sid] for sid in reales),
                "etiqueta": "",
                "nota": "",
            })

    if os.path.exists(ruta_etiquetas):
        print(f"Aviso: {ruta_etiquetas} ya existe, no se sobreescribe (bórralo si quieres regenerarlo).")
    else:
        pd.DataFrame(filas_plantilla).to_csv(ruta_etiquetas, index=False)
        print(f"Plantilla guardada en {ruta_etiquetas}")
    print(f"Revisión guardada en {ruta_revision} ({len(palabras)} palabras)")


def analizar_etiquetas(ruta_etiquetas=RUTA_ETIQUETAS, ruta_separabilidad=RUTA_SEPARABILIDAD):
    etiquetas = pd.read_csv(ruta_etiquetas)
    etiquetas["etiqueta"] = etiquetas["etiqueta"].fillna("").str.strip().str.lower()
    sin_etiqueta = (etiquetas["etiqueta"] == "").sum()
    if sin_etiqueta:
        print(f"Aviso: {sin_etiqueta} palabras sin etiqueta -- se excluyen")

    sep = pd.read_csv(ruta_separabilidad)
    df = sep.merge(etiquetas[["palabra", "etiqueta"]], on="palabra")
    df = df[df["etiqueta"] != ""].dropna(subset=["mejora_sobre_base"])

    print("=== Palabras por etiqueta ===")
    print(etiquetas.loc[etiquetas["etiqueta"] != "", "etiqueta"].value_counts().to_string())

    print("\n=== Mejora de k-NN sobre la clase mayoritaria, por etiqueta y modelo ===")
    resumen = df.groupby(["etiqueta", "modelo"]).agg(
        n=("palabra", "size"),
        mejora_mediana=("mejora_sobre_base", "median"),
        pct_separable=("p_accuracy", lambda s: (s < 0.05).mean()),
    )
    print(resumen.to_string())

    print("\n=== ¿Los embeddings separan más las particiones 'sentido' que las 'tema'? "
          "(Mann-Whitney, una cola) ===")
    for modelo, grupo in df.groupby("modelo"):
        a = grupo.loc[grupo["etiqueta"] == "sentido", "mejora_sobre_base"]
        b = grupo.loc[grupo["etiqueta"] == "tema", "mejora_sobre_base"]
        if len(a) < 3 or len(b) < 3:
            print(f"  {modelo}: muy pocas palabras (sentido={len(a)}, tema={len(b)})")
            continue
        _, p = mannwhitneyu(a, b, alternative="greater")
        print(f"  {modelo}: mediana sentido={a.median():.3f} vs tema={b.median():.3f}, "
              f"p={p:.4g} (n={len(a)} vs {len(b)})")
    return df


if __name__ == "__main__":
    accion = sys.argv[1] if len(sys.argv) > 1 else "generar"
    if accion == "generar":
        generar_revision()
    elif accion == "analizar":
        analizar_etiquetas()
    else:
        print("Uso: python revisar_glosas_juez.py [generar|analizar]")