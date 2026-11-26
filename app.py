import os, re, csv, datetime
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", "/content/drive/MyDrive/news_classifier_v2")
FB_FILE = os.getenv("FEEDBACK_FILE", os.path.join(BASE, "news_feedback.csv"))

with open(os.path.join(BASE, "index.html"), encoding="utf-8") as f:
    HTML_PAGE = f.read()

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
mdl.to(device).eval()
id2label = mdl.config.id2label

PASHTO_CHARS = set("ځڅډړږښټڼېۍګ")

def guess_lang(t):
    if any(c in PASHTO_CHARS for c in t):
        return "Pashto"
    if re.search(r"[\u0600-\u06FF]", t):
        return "Urdu"
    return "English"

def run_model(texts, threshold):
    enc = tok(texts, truncation=True, padding=True, max_length=64,
              return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(mdl(**enc).logits, dim=-1).cpu().numpy()
    results = []
    for t, p in zip(texts, probs):
        i = int(p.argmax())
        conf = float(p[i])
        results.append({
            "text": t,
            "label": id2label[i],
            "confidence": round(conf, 4),
            "unsure": conf < threshold,
            "language": guess_lang(t),
            "scores": {id2label[j]: round(float(p[j]), 4) for j in range(len(p))},
        })
    return results

class Item(BaseModel):
    text: str
    threshold: float = 0.5

class BatchItem(BaseModel):
    texts: List[str]
    threshold: float = 0.5

class Feedback(BaseModel):
    text: str
    predicted: str
    correct: str

app = FastAPI(title="Connected Karak News AI")

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

@app.get("/health")
def health():
    return {"status": "ok", "labels": list(id2label.values()), "device": device}

@app.post("/predict")
def predict_one(item: Item):
    text = item.text.strip()
    if not text:
        raise HTTPException(400, "Empty text")
    return run_model([text], item.threshold)[0]

@app.post("/predict_batch")
def predict_batch(item: BatchItem):
    texts = [t.strip() for t in item.texts if t.strip()][:200]
    if not texts:
        raise HTTPException(400, "No text")
    return {"count": len(texts), "results": run_model(texts, item.threshold)}

@app.post("/feedback")
def feedback(fb: Feedback):
    new_file = not os.path.exists(FB_FILE)
    with open(FB_FILE, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "text", "predicted", "correct"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                    fb.text, fb.predicted, fb.correct])
    return {"saved": True}
