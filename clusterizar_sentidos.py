# %% [markdown]
# # Clustering exploratorio de la estructura de embeddings por palabra (parte 2 de 2)
#
# **Alcance del análisis, léelo antes de interpretar resultados:** se probó
# quitar componentes principales dominantes (whitening) y medir silhouette
# score para intentar distinguir separaciones "reales" (sentido léxico) de
# separaciones espurias (ruido de registro/tema/longitud del tweet) en
# palabras de control conocidas. **No se encontró ninguna combinación de
# parámetros que lograra esa distinción** — las distribuciones de silhouette
# de splits en palabras monosémicas y polisémicas de control se superponen
# casi por completo (ver calibración, sección 3). Por lo tanto, este notebook
# ya NO reporta "polisemia" como verdad determinada — reporta **estructura de
# clusters observada en los embeddings de cada modelo** (cuántos grupos
# separables produce, qué tan compactos son), sin asumir que esa estructura
# corresponda a sentidos léxicos reales. Las columnas de salida se llaman
# `k_clusters` y `presenta_multiples_clusters` (no `k_sentidos`/`es_polisemica`)
# a propósito, para no sobre-afirmar lo que el método puede validar.
#
# Lee los `.npz` guardados por `extraer_embeddings.py` — no necesita GPU ni
# `transformers`, solo numpy/pandas/umap/hdbscan/sklearn.
#
# Requiere: umap-learn, hdbscan, scikit-learn, pandas, numpy
# Ejecuta: pip install umap-learn hdbscan scikit-learn pandas numpy --break-system-packages

# %%
import os
import csv
import unicodedata
import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import defaultdict

# %% [markdown]
# ## Configuración

# %%
MODELOS = ["mex_state", "mex_large", "beto", "bertin", "robertuito"]
CARPETA_EMBEDDINGS = "embeddings"

UMAP_N_COMPONENTS = 12
HDBSCAN_MIN_SAMPLES = 3
MIN_OCURRENCIAS_PARA_CLUSTER = 8   # palabras con menos ocurrencias no se agrupan

# %% [markdown]
# ## 0. Cargar embeddings guardados
#
# Lee el `.npz` de un modelo y devuelve `(embeddings, metadata)`, donde
# `metadata` es una lista de dicts con al menos la clave `palabra` por fila
# (más `id_tweet`/`region` si el `.npz` las incluye). Si guardaste el tuyo con
# `extraer_embeddings.py` las claves son `embeddings`/`words`/`id_tweets` — si
# usaste tu propio notebook con otras claves (ej. `region` incluida), esta
# función las toma tal cual vengan, solo mapea `words` → `palabra`.

# %%
def cargar_embeddings(nombre_modelo, carpeta=CARPETA_EMBEDDINGS):
    ruta = os.path.join(carpeta, f"{nombre_modelo}_embeddings.npz")
    data = np.load(ruta, allow_pickle=True)
    emb = data["embeddings"]
    claves_meta = [k for k in data.files if k != "embeddings"]
    n = emb.shape[0]
    metadata = []
    for i in range(n):
        fila = {}
        for clave in claves_meta:
            nombre_campo = "palabra" if clave == "words" else clave
            fila[nombre_campo] = data[clave][i]
        metadata.append(fila)
    return emb, metadata

# Verifica una vez qué claves trae tu .npz antes de seguir:
# _, _w = cargar_embeddings(MODELOS[0])
# print(_w[0])

# %% [markdown]
# ## 1. Clustering exploratorio por palabra (HDBSCAN)
#
# **`min_cluster_size` adaptativo, no fijo.** Un valor fijo (ej. 3) es razonable
# para una palabra con 20 ocurrencias, pero para una con 1,000+ ocurrencias es
# absurdamente permisivo — HDBSCAN encuentra decenas de micro-agrupaciones que
# son ruido estadístico (esto se confirmó con la corrida completa: 0% de
# palabras con k=1 en los 5 modelos, y k máximo de hasta 281). Este ajuste
# sigue siendo válido y necesario independientemente de la limitación de
# fondo descrita arriba — evita resultados absurdos, aunque no garantiza que
# el k resultante sea "sentidos reales".
#
# **`quitar_componentes_dominantes` (whitening) queda disponible pero SIN
# evidencia de que resuelva el problema de fondo.** Se probó como corrección
# contra anisotropía de embeddings de BERT, pero la calibración (sección 3)
# mostró que ni el número de clusters ni el silhouette score distinguen
# separaciones reales de espurias, con o sin esta corrección. Se deja
# implementada por si se retoma en el futuro con otro enfoque de validación,
# pero el valor final usado en la orquestación (sección 4) es 0.

# %%
MCS_FRACCION = 0.05   # min_cluster_size ≈ 5% de las ocurrencias de la palabra
MCS_MINIMO = 5
MCS_MAXIMO = 50

def calcular_min_cluster_size(n, fraccion=MCS_FRACCION, minimo=MCS_MINIMO, maximo=MCS_MAXIMO):
    return int(np.clip(round(fraccion * n), minimo, maximo))


def quitar_componentes_dominantes(X, n_componentes=1):
    """Centra y quita las primeras n_componentes principales de las ocurrencias
    de UNA palabra. La dirección de mayor varianza en embeddings contextuales
    de BERT suele ser genérica (anisotropía: registro, longitud, tema) y no
    semántica — quitarla antes de clusterizar evita que HDBSCAN la confunda
    con sentidos distintos."""
    n_comp = min(n_componentes, X.shape[0] - 1, X.shape[1])
    media = X.mean(axis=0)
    X_centrado = X - media
    if n_comp < 1:
        return X_centrado
    pca = PCA(n_components=n_comp)
    pca.fit(X_centrado)
    proyeccion = pca.transform(X_centrado) @ pca.components_
    return X_centrado - proyeccion


def _hdbscan_labels(embeddings, min_cluster_size, min_samples, n_componentes_quitar=0):
    """Corre UMAP+HDBSCAN sobre los embeddings de una palabra y devuelve
    (labels, silhouette). labels: array de labels (uno por ocurrencia, -1 =
    ruido). silhouette: score de separación entre clusters (None si k<2 o muy
    pocos puntos no-ruido) -- útil para distinguir splits reales de espurios,
    algo que el número de clusters solo no puede decirnos. Si
    n_componentes_quitar>0, primero centra y quita esa(s) componente(s)
    principal(es) dominante(s) (ver quitar_componentes_dominantes)."""
    X = np.vstack(embeddings)
    if n_componentes_quitar > 0:
        X = quitar_componentes_dominantes(X, n_componentes_quitar)
    n = len(embeddings)
    n_components = min(UMAP_N_COMPONENTS, n - 2)
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    X_red = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(X_red)

    k = len(set(labels) - {-1})
    mask = labels != -1
    if k >= 2 and mask.sum() >= k + 1:
        silhouette = silhouette_score(X_red[mask], labels[mask])
    else:
        silhouette = None
    return labels, silhouette


def clusterizar_palabra(embeddings, min_cluster_size=None, min_samples=None, n_componentes_quitar=0):
    """Dado un array de embeddings (n_ocurrencias, dim) para UNA palabra en UN modelo,
    devuelve (k_clusters, pct_ruido, min_cluster_size_usado, silhouette). Si no
    se pasa min_cluster_size explícito, se calcula de forma adaptativa según
    `n` (ver calcular_min_cluster_size) — NO se usa un valor fijo global."""
    n = len(embeddings)
    if n < MIN_OCURRENCIAS_PARA_CLUSTER:
        return None, None, None, None  # no evaluada, muy pocas ocurrencias

    mcs = min_cluster_size if min_cluster_size is not None else calcular_min_cluster_size(n)
    ms = min_samples if min_samples is not None else HDBSCAN_MIN_SAMPLES

    labels, silhouette = _hdbscan_labels(embeddings, mcs, ms, n_componentes_quitar=n_componentes_quitar)
    k = len(set(labels) - {-1})
    pct_ruido = (labels == -1).mean() * 100
    return k, pct_ruido, mcs, silhouette


def inspeccionar_ocurrencias(palabra, embeddings_dict, tweets_dict, min_cluster_size=None, min_samples=None, n_componentes_quitar=0):
    """Devuelve un DataFrame (id_tweet, cluster_id) para revisar caso por caso
    a qué tweet corresponde cada ocurrencia y en qué cluster cayó (cluster,
    no "sentido" — ver limitación en el encabezado del notebook)."""
    embs = embeddings_dict[palabra]
    tids = tweets_dict[palabra]
    mcs = min_cluster_size if min_cluster_size is not None else calcular_min_cluster_size(len(embs))
    ms = min_samples if min_samples is not None else HDBSCAN_MIN_SAMPLES
    labels, _ = _hdbscan_labels(embs, mcs, ms, n_componentes_quitar=n_componentes_quitar)
    return pd.DataFrame({"id_tweet": tids, "cluster_id": labels})

# %% [markdown]
# ## 2. Filtro de palabras funcionales (stopwords)
#
# Palabras funcionales de alta frecuencia (preposiciones, artículos...) se
# fragmentan en decenas de micro-clusters espurios si no se excluyen — no
# tiene sentido preguntarse si son "polisémicas" en el sentido léxico que
# interesa aquí. Se excluyen del clustering (quedan registradas con
# `excluida_stopword=True` para trazabilidad, no se eliminan silenciosamente).

# %%
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    nltk.download("stopwords", quiet=True)
    STOPWORDS_ES = set(nltk_stopwords.words("spanish"))
except Exception:
    # Respaldo si no hay conexión para nltk.download
    STOPWORDS_ES = {
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
        "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le",
        "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre",
        "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante",
        "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e",
        "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él",
        "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella",
        "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu",
        "tus", "ellas", "nosotras", "vosotros", "vosotras", "os", "esos", "esas", "soy",
        "eres", "es", "somos", "sois", "son", "era", "eras", "fui",
    }
print(f"Stopwords cargadas: {len(STOPWORDS_ES)}")

# %% [markdown]
# ## 3. Calibración/diagnóstico exploratorio — correr ANTES del pipeline completo
#
# **Resultado ya conocido: ninguna combinación aquí "valida" polisemia real**
# (ver limitación en el encabezado). Esta sección se deja como diagnóstico de
# comportamiento del pipeline (qué tan sensible es a los parámetros, cómo se
# comparan los 5 modelos entre sí) — no como búsqueda de la config "correcta".
# Carga las palabras de calibración directamente de los `.npz` ya guardados.

# %%
PALABRAS_CALIBRACION_POLISEMICAS = {"banco", "gato", "planta", "sierra", "cola", "papa", "vela", "muñeca"}
PALABRAS_CALIBRACION_MONOSEMICAS = {"bicicleta", "computadora", "cumpleaños", "desayuno",
                                      "aeropuerto", "estudiante", "diciembre", "elefante"}
PALABRAS_CALIBRACION = PALABRAS_CALIBRACION_POLISEMICAS | PALABRAS_CALIBRACION_MONOSEMICAS

embeddings_calib_por_modelo = {}
tweets_calib_por_modelo = {}

for nombre_modelo in MODELOS:
    emb, w = cargar_embeddings(nombre_modelo)
    # normaliza a NFC -- "ñ" puede venir como un solo carácter o como "n" + tilde
    # combinada; se ven idénticas pero no son el mismo string en Python
    palabras_arr = np.array([unicodedata.normalize("NFC", str(d["palabra"])) for d in w])
    tweets_arr = np.array([d.get("id_tweet", d.get("tweet_id")) for d in w])

    emb_dict, tw_dict = defaultdict(list), defaultdict(list)
    for palabra in PALABRAS_CALIBRACION:
        palabra_nfc = unicodedata.normalize("NFC", palabra)
        idx = np.where(palabras_arr == palabra_nfc)[0]
        if len(idx) == 0:
            print(f"  [{nombre_modelo}] '{palabra}' no está en el .npz guardado — sáltala o revisa el nombre")
            continue
        emb_dict[palabra] = list(emb[idx])
        tw_dict[palabra] = list(tweets_arr[idx])
    embeddings_calib_por_modelo[nombre_modelo] = emb_dict
    tweets_calib_por_modelo[nombre_modelo] = tw_dict

resultados_calibracion = []
CONFIGS_CALIBRACION = ["adaptativo", 3, 5, 10]
COMPONENTES_A_PROBAR = [0, 1, 2, 3, 5]

for nombre_modelo in MODELOS:
    for palabra, embs in embeddings_calib_por_modelo[nombre_modelo].items():
        tipo_esperado = "polisémica" if palabra in PALABRAS_CALIBRACION_POLISEMICAS else "monosémica"
        for config in CONFIGS_CALIBRACION:
            mcs_explicito = None if config == "adaptativo" else config
            for n_comp in COMPONENTES_A_PROBAR:
                k, pct_ruido, mcs_usado, silhouette = clusterizar_palabra(
                    embs, min_cluster_size=mcs_explicito, min_samples=HDBSCAN_MIN_SAMPLES,
                    n_componentes_quitar=n_comp,
                )
                resultados_calibracion.append({
                    "modelo": nombre_modelo, "palabra": palabra, "tipo_esperado": tipo_esperado,
                    "n_ocurrencias": len(embs), "config": config, "min_cluster_size_usado": mcs_usado,
                    "n_componentes_quitar": n_comp, "k_clusters": k, "pct_ruido": pct_ruido,
                    "silhouette": silhouette,
                })

calibracion_df = pd.DataFrame(resultados_calibracion)

# En vez de asumir un umbral de silhouette, comparamos la distribución real:
# separaciones "reales" (polisémicas con k>=2) vs. "espurias" (monosémicas con
# k>=2) -- si hay una brecha clara, ahí está el corte a usar; si se solapan,
# el silhouette tampoco distingue y hay que pensar en otra cosa.
splits_reales = calibracion_df[(calibracion_df["tipo_esperado"] == "polisémica") & (calibracion_df["k_clusters"] >= 2)]
splits_espurios = calibracion_df[(calibracion_df["tipo_esperado"] == "monosémica") & (calibracion_df["k_clusters"] >= 2)]

print("=== Distribución de silhouette en splits de polisémicas (se espera que sean reales) ===")
print(splits_reales.groupby("n_componentes_quitar")["silhouette"].describe())

print("\n=== Distribución de silhouette en splits de monosémicas (se espera que sean espurios) ===")
print(splits_espurios.groupby("n_componentes_quitar")["silhouette"].describe())

# %% [markdown]
# ### Inspección caso por caso (opcional)

# %%
# Ejemplo — ajusta modelo, palabra y parámetros elegidos:
# inspeccionar_ocurrencias(
#     "diciembre",
#     embeddings_calib_por_modelo["beto"],
#     tweets_calib_por_modelo["beto"],
#     min_cluster_size=5, min_samples=HDBSCAN_MIN_SAMPLES, n_componentes_quitar=1,
# )

# %% [markdown]
# ## 4. Orquestación — correr los 5 modelos sobre el corpus completo
#
# `N_COMPONENTES_QUITAR_FINAL = 0`: el whitening no demostró distinguir
# separaciones reales de espurias (sección 3), así que no se usa una
# corrección sin evidencia de que funcione — se mantiene el pipeline simple.
# `presenta_multiples_clusters` describe estructura de embeddings observada,
# no polisemia validada (ver limitación en el encabezado del notebook).
# Guarda incrementalmente (append + flush por palabra) en un CSV nuevo, y es
# reanudable si el proceso se cae a mitad.

# %%
N_COMPONENTES_QUITAR_FINAL = 0    # sin evidencia de que el whitening ayude (sección 3) -- se deja en 0
RESULTADOS_CSV = "resultados_clustering_v2.csv"
CAMPOS_RESULTADOS = ["word_normalized", "modelo", "n_ocurrencias", "min_cluster_size_usado",
                      "n_componentes_quitar", "k_clusters", "pct_ruido", "presenta_multiples_clusters", "excluida_stopword"]
UMBRAL_RUIDO_NO_DETERMINADO = 50  # % de ruido a partir del cual el cluster no es confiable / no se reporta


def cargar_procesados(ruta=RESULTADOS_CSV):
    """Devuelve el set de (modelo, word_normalized) ya procesados, leyendo el
    CSV si existe. Tolera una última línea corrupta (ej. si el proceso se cayó
    a mitad de una escritura)."""
    if not os.path.exists(ruta):
        return set()
    try:
        df_previo = pd.read_csv(ruta, on_bad_lines="skip")
    except TypeError:  # pandas viejo sin on_bad_lines
        df_previo = pd.read_csv(ruta, error_bad_lines=False)
    return set(zip(df_previo["modelo"], df_previo["word_normalized"]))


def clusterizar_todas_las_palabras_resumible(nombre_modelo, min_cluster_size=None, min_samples=None,
                                               n_componentes_quitar=N_COMPONENTES_QUITAR_FINAL, ruta=RESULTADOS_CSV):
    """Igual que clusterizar_palabra, pero recorre TODAS las palabras del
    modelo y guarda el resultado de CADA una apenas termina (append + flush a
    `ruta`), saltándose las que ya estén en el archivo — así, si el proceso se
    cae a mitad, vuelves a correr esta misma función y retoma donde se quedó."""
    emb, w = cargar_embeddings(nombre_modelo)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    palabras_arr = np.array([d["palabra"] for d in w])
    palabras_unicas = np.unique(palabras_arr)

    procesados = cargar_procesados(ruta)
    escribir_header = not os.path.exists(ruta)
    ya_hechas = sum(1 for p in palabras_unicas if (nombre_modelo, p) in procesados)
    print(f"--- {nombre_modelo}: {ya_hechas}/{len(palabras_unicas)} palabras ya procesadas, retomando ---")

    with open(ruta, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS_RESULTADOS)
        if escribir_header:
            writer.writeheader()
            f.flush()

        for i, palabra in enumerate(palabras_unicas):
            if (nombre_modelo, palabra) in procesados:
                continue

            if palabra in STOPWORDS_ES:
                writer.writerow({
                    "word_normalized": palabra, "modelo": nombre_modelo,
                    "n_ocurrencias": None, "min_cluster_size_usado": None,
                    "n_componentes_quitar": None, "k_clusters": None, "pct_ruido": None,
                    "presenta_multiples_clusters": None, "excluida_stopword": True,
                })
                f.flush()
                continue

            idx = np.where(palabras_arr == palabra)[0]
            embs_palabra = emb[idx]
            n = len(embs_palabra)

            if n < MIN_OCURRENCIAS_PARA_CLUSTER:
                k, pct_ruido, mcs_usado, presenta_multiples_clusters = None, None, None, None
            else:
                mcs = min_cluster_size if min_cluster_size is not None else calcular_min_cluster_size(n)
                ms = min_samples if min_samples is not None else HDBSCAN_MIN_SAMPLES
                labels, _silhouette = _hdbscan_labels(embs_palabra, mcs, ms, n_componentes_quitar=n_componentes_quitar)
                k = len(set(labels) - {-1})
                pct_ruido = (labels == -1).mean() * 100
                mcs_usado = mcs
                presenta_multiples_clusters = None if pct_ruido > UMBRAL_RUIDO_NO_DETERMINADO else (k > 1)

            writer.writerow({
                "word_normalized": palabra, "modelo": nombre_modelo,
                "n_ocurrencias": n, "min_cluster_size_usado": mcs_usado,
                "n_componentes_quitar": n_componentes_quitar if n >= MIN_OCURRENCIAS_PARA_CLUSTER else None,
                "k_clusters": k, "pct_ruido": pct_ruido,
                "presenta_multiples_clusters": presenta_multiples_clusters, "excluida_stopword": False,
            })
            f.flush()  # persiste en disco apenas termina esta palabra, no al final del loop

            if i % 500 == 0:
                print(f"  [{nombre_modelo}] {i}/{len(palabras_unicas)} palabras procesadas")

    print(f"--- {nombre_modelo}: completado ---")


for nombre_modelo in MODELOS:
    clusterizar_todas_las_palabras_resumible(nombre_modelo)

resultados_df = pd.read_csv(RESULTADOS_CSV)
print(resultados_df.head(20))
print(f"\nTotal de filas (palabra x modelo): {len(resultados_df)}")