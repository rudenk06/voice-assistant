import numpy as np
import sqlite3
import faiss
import json
import time
from src.rag.embedder import Embedder

ground_truth = {
    # Чанк 0 — заведующий
    "кто заведующий кафедрой": 0,
    "кто руководит кафедрой": 0,
    "кто завкаф": 0,
    # Чанк 1 — основание
    "когда основана кафедра": 1,
    "в каком году создали кафедру": 1,
    "история кафедры": 1,
    # Чанк 2 — направления
    "какие научные направления": 2,
    "чем занимается кафедра": 2,
    "направления исследований": 2,
    # Чанк 3 — Малинина Мария Анатольевна
    "как связаться с Малининой Марией Анатольевной?": 3,
    "почта Малининой Марии Анатольевны": 3,
    "где кабинет Малининой Марии Анатольевны?": 3,
    # Чанк 4 — Митрофанов Евгений Павлович
    "как связаться с Митрофановым Евгением Павловичем?": 4,
    "почта Митрофанова Евгения Павловича": 4,
    "где найти Евгения Митрофанова Павловича?": 4,
    # Чанк 5 — Попова Светлана Владимировна
    "как связаться с Поповой Светланой Владимировной?": 5,
    "почта Поповы Светланы Владимировны": 5,
    "где найти Попову Светлану Владимировну?": 5,
    # Чанк 6 — Митрофанова Ольга Александровна
    "как связаться с Митрофановой Ольгой Александровой": 6,
    "почта Митрофановы Ольги Александровны": 6,
    "где найти Митрофанову Ольгу Александровну?": 6,
}

model_path = "data/models/rubert-tiny2-int8"
model_name = "rubert-tiny2"

embedder = Embedder(model_path)
embedder.load()
index = faiss.read_index("data/index/faiss.index")
conn = sqlite3.connect("data/index/chunks.db")

results = []
total_time = 0

for query, correct_id in ground_truth.items():
    start = time.time()
    vec = embedder.embed([query])
    elapsed = time.time() - start
    total_time += elapsed

    scores, indices = index.search(vec, 3)
    top1 = int(indices[0][0])
    top3 = [int(i) for i in indices[0]]
    top1_score = float(scores[0][0])

    hit1 = 1 if top1 == correct_id else 0

    if correct_id in top3:
        rank = top3.index(correct_id) + 1
        rr = 1.0 / rank
    else:
        rr = 0.0

    hit3 = 1 if correct_id in top3 else 0

    row = conn.execute("SELECT text FROM chunks WHERE embedding_id=?", (top1,)).fetchone()
    found = row[0][:70] if row else "NOT FOUND"

    results.append({
        "query": query,
        "correct": correct_id,
        "found": top1,
        "score": round(top1_score, 4),
        "hit1": hit1,
        "rr": round(rr, 4),
        "hit3": hit3,
        "ms": round(elapsed * 1000, 1),
    })

    status = "+" if hit1 else "-"
    print(f"{status} [{top1_score:.3f}] '{query}' -> chunk {top1} ({elapsed*1000:.0f}ms)")
    print(f"       {found}")

p1 = sum(r["hit1"] for r in results) / len(results)
mrr = sum(r["rr"] for r in results) / len(results)
r3 = sum(r["hit3"] for r in results) / len(results)
avg_score = sum(r["score"] for r in results) / len(results)
avg_ms = total_time / len(results) * 1000

print(f"\n{'='*50}")
print(f"Model:        {model_name}")
print(f"Questions:    {len(results)}")
print(f"{'='*50}")
print(f"Precision@1:  {p1:.3f}")
print(f"MRR:          {mrr:.3f}")
print(f"Recall@3:     {r3:.3f}")
print(f"Avg Score:    {avg_score:.4f}")
print(f"Avg Latency:  {avg_ms:.1f} ms")
print(f"{'='*50}")

report = {
    "model": model_name,
    "precision_at_1": round(p1, 4),
    "mrr": round(mrr, 4),
    "recall_at_3": round(r3, 4),
    "avg_score": round(avg_score, 4),
    "avg_latency_ms": round(avg_ms, 1),
    "details": results,
}
with open(f"benchmark_{model_name}.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\nSaved to benchmark_{model_name}.json")
embedder.unload()
