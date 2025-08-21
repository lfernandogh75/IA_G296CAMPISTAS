# ==========================================
# train_chatbot.py
# Entrena un clasificador de intenciones y guarda:
#  - models/chatbot_model.pkl  (modelo TF-IDF + SGDClassifier)
#  - models/responses.json     (tag -> lista de respuestas)
# ==========================================

import os              # Manejo de rutas/carpetas
import re              # Expresiones regulares para normalización
import json            # Cargar/guardar JSON (intents y respuestas)
import argparse        # CLI: recibir parámetros --intents, --model, etc.
from typing import List, Dict, Tuple  # Tipado opcional para claridad
import pickle          # Guardar/cargar el modelo entrenado en .pkl

# sklearn: pipeline + modelo lineal + vectorizador TF-IDF
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# ------- Normalización: quitar acentos y dejar minúsculas/solo letras -------
def _fallback_normalize(text: str) -> str:
    text = text.lower()                                # Minúsculas
    trans = str.maketrans("áéíóúüñ", "aeiouun")       # Mapa simple acentos->ASCII
    text = text.translate(trans)                       # Aplicar mapa
    text = re.sub(r"[^a-z\s]", " ", text)              # Dejar solo letras y espacios
    return re.sub(r"\s+", " ", text).strip()           # Colapsar espacios

# Intentar usar unidecode si está instalado (mejor quita diacríticos)
try:
    from unidecode import unidecode
    def normalize(text: str) -> str:
        text = text.lower()                            # Minúsculas
        text = unidecode(text)                         # Quitar tildes/diacríticos
        text = re.sub(r"[^a-z\s]", " ", text)          # Mantener solo letras/espacios
        return re.sub(r"\s+", " ", text).strip()       # Normalizar espacios
except Exception:
    normalize = _fallback_normalize                    # Fallback si no hay unidecode

# ------- Cargar intents (JSON con intents -> patrones/respuestas) -------
def load_intents(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:       # Abrir archivo JSON
        data = json.load(f)                            # Parsear a dict
    assert "intents" in data and isinstance(data["intents"], list), "intents.json inválido."
    return data

# ------- Construir dataset (X: patrones, y: tags) -------
def build_dataset(intents: List[Dict]) -> Tuple[List[str], List[str]]:
    X, y = [], []                                      # X: textos, y: etiquetas/tag
    for it in intents:                                 # Recorrer cada intent
        tag = it["tag"]                                # Etiqueta de esa intención
        for p in it["patterns"]:                       # Patrones de ejemplo (textos)
            X.append(p)                                # Agregar texto a X
            y.append(tag)                              # Agregar tag correspondiente a y
    return X, y

# ------- Guardar mapa de respuestas por tag en JSON -------
def save_responses(intents: List[Dict], out_path: str):
    mapping = {it["tag"]: it["responses"] for it in intents}  # tag -> lista de respuestas
    os.makedirs(os.path.dirname(out_path), exist_ok=True)      # Crear carpeta si no existe
    with open(out_path, "w", encoding="utf-8") as f:           # Abrir archivo de salida
        json.dump(mapping, f, ensure_ascii=False, indent=2)    # Guardar formato legible

# ------- Punto de entrada principal -------
def main():
    # Definir argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Entrena el modelo de intenciones para el chatbot.")
    parser.add_argument("--intents", required=True, help="Ruta al intents.json")  # Obligatorio
    parser.add_argument("--model", default="models/chatbot_model.pkl", help="Ruta de salida del modelo .pkl")
    parser.add_argument("--responses", default="models/responses.json", help="Ruta de salida de respuestas por tag")
    parser.add_argument("--max-iter", type=int, default=2000, help="Iteraciones del SGDClassifier")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--min-df", type=int, default=1, help="min_df del TfidfVectorizer (filtra rarezas)")
    parser.add_argument("--ngram-max", type=int, default=2, help="n-grama máximo (1=unigramas, 2=bigramas, etc.)")
    args = parser.parse_args()                          # Parsear CLI

    data = load_intents(args.intents)                   # Cargar intents.json
    intents = data["intents"]                           # Lista de intents
    X, y = build_dataset(intents)                       # Armar X/y de entrenamiento

    # Crear pipeline: TF-IDF (con normalización custom) + SGDClassifier (log_loss => probabilidades)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, args.ngram_max), min_df=args.min_df)),
        ("clf", SGDClassifier(loss="log_loss", max_iter=args.max_iter, tol=1e-4, random_state=args.random_state))
    ])

    model = pipe.fit(X, y)                              # Entrenar modelo con datos X/y

    os.makedirs(os.path.dirname(args.model), exist_ok=True)    # Asegurar carpeta models/
    with open(args.model, "wb") as f:                           # Abrir archivo .pkl binario
        pickle.dump(model, f)                                   # Guardar pipeline completo

    save_responses(intents, args.responses)             # Guardar respuestas por tag

    acc = model.score(X, y)                             # Accuracy sobre train (referencial)
    print(">> Entrenamiento completado")
    print(f"   - Patrones de entrenamiento: {len(X)}")
    print(f"   - Clases: {sorted(set(y))}")
    print(f"   - Accuracy (sobre train): {acc:.3f}")
    print(f"   - Modelo: {args.model}")
    print(f"   - Respuestas: {args.responses}")

# Ejecutar main() si este archivo se corre como script
if __name__ == "__main__":
    main()
