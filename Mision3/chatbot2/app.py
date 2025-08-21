# ==========================================
# app.py
# Carga el modelo entrenado y expone un chatbot por Flask
# ==========================================
import os
import re
import json
import random
import argparse
import pickle
from typing import Dict, List
from flask import Flask, request, jsonify, render_template

def _fallback_normalize(text: str) -> str:
    text = text.lower()
    trans = str.maketrans("áéíóúüñ", "aeiouun")
    text = text.translate(trans)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

try:
    from unidecode import unidecode
    def normalize(text: str) -> str:
        text = text.lower()
        text = unidecode(text)
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
except Exception:
    normalize = _fallback_normalize

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_responses(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_reply(model, message: str, responses: Dict[str, List[str]], threshold: float = 0.45) -> str:
    probs = model.predict_proba([message])[0]
    classes = model.classes_
    top_idx = probs.argmax()
    top_tag = classes[top_idx]
    top_conf = float(probs[top_idx])
    if top_conf < threshold or top_tag not in responses:
        return "No estoy seguro de entender. ¿Puedes darme más detalles?"
    return random.choice(responses[top_tag])

def main():
    parser = argparse.ArgumentParser(description="Servidor Flask para el chatbot.")
    parser.add_argument("--model", default="models/chatbot_model.pkl", help="Ruta al modelo .pkl entrenado")
    parser.add_argument("--responses", default="models/responses.json", help="Ruta al JSON de respuestas")
    parser.add_argument("--host", default="0.0.0.0", help="Host de Flask")
    parser.add_argument("--port", type=int, default=5000, help="Puerto de Flask")
    parser.add_argument("--threshold", type=float, default=0.45, help="Umbral de confianza")
    args = parser.parse_args()

    app = Flask(__name__)  # templates/ y static/ son detectadas por defecto

    model = load_model(args.model)
    responses = load_responses(args.responses)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/chat")
    def api_chat():
        data = request.get_json(silent=True) or {}
        msg = (data.get("message") or "").strip()
        if not msg:
            return jsonify({"reply": "Escribe un mensaje para comenzar."})
        reply = get_reply(model, msg, responses, threshold=args.threshold)
        return jsonify({"reply": reply})

    print(f">> Servidor listo en http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)

if __name__ == "__main__":
    main()
