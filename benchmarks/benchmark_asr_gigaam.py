import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

torch.serialization.add_safe_globals([DictConfig, ListConfig, OmegaConf])
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import gigaam
import json
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import jiwer

phrases = [
    # Общие вопросы
    "кто заведующий кафедрой",
    "кто руководит кафедрой",
    "когда основана кафедра",
    "в каком году создали кафедру",
    "какие научные направления",
    "чем занимается кафедра",
    "направления исследований",
    "как связаться с Малининой Марией Анатольевной",
    "почта Малининой Марии Анатольевны",
    "где кабинет Малининой Марии Анатольевны",
    "как связаться с Митрофановым Евгением Павловичем ",
    "почта Митрофанова Евгения Павловича",
    "где найти Митрофанова Евгения Павловича",
    "как связаться с Поповой Светланой Владимировной",
    "почта Поповой Светланы Владимировны",
    "где найти Попову Светлану Владимировну",
    "как связаться с Митрофановой Ольгой Александровной",
    "почта Митрофановой Ольги Александровны",
    "где найти Митрофанову Ольгу Александровну",
    "почта заведующего кафедрой",
    "как связаться с заведующим",
]

RECORD_SECONDS = 4
SAMPLE_RATE = 16000

print("Загрузка модели GigaAM...")
model = gigaam.load_model("ctc")
print("Модель загружена!\n")

results = []

for i, phrase in enumerate(phrases):
    print(f"\n[{i+1}/{len(phrases)}]")
    print(f"  Скажи: \"{phrase}\"")
    input("  Нажми Enter и говори...")

    # Запись
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    # Сохраняем во временный файл
    sf.write("temp_audio.wav", audio, SAMPLE_RATE)

    # Распознавание + замер времени
    start = time.time()
    recognized = model.transcribe("temp_audio.wav")
    elapsed = time.time() - start

    # Нормализация
    ref = phrase.lower().strip()
    hyp = recognized.lower().strip() if recognized else ""

    word_wer = jiwer.wer(ref, hyp) if hyp else 1.0
    char_cer = jiwer.cer(ref, hyp) if hyp else 1.0

    match = "ВЕРНО" if ref == hyp else "ОШИБКА"

    print(f"  Эталон:    \"{ref}\"")
    print(f"  Распознал: \"{hyp}\"")
    print(f"  WER: {word_wer:.2%}  CER: {char_cer:.2%}  Время: {elapsed*1000:.0f}ms  [{match}]")

    results.append({
        "phrase_id": i + 1,
        "reference": ref,
        "hypothesis": hyp,
        "exact_match": ref == hyp,
        "wer": round(word_wer, 4),
        "cer": round(char_cer, 4),
        "latency_ms": round(elapsed * 1000, 1),
    })

# Итоговые метрики
all_refs = [r["reference"] for r in results]
all_hyps = [r["hypothesis"] if r["hypothesis"] else " " for r in results]

total_wer = jiwer.wer(all_refs, all_hyps)
total_cer = jiwer.cer(all_refs, all_hyps)

try:
    output = jiwer.process_words(all_refs, all_hyps)
    substitutions = output.substitutions
    deletions = output.deletions
    insertions = output.insertions
    hits = output.hits
except AttributeError:
    measures = jiwer.compute_measures(all_refs, all_hyps)
    substitutions = measures['substitutions']
    deletions = measures['deletions']
    insertions = measures['insertions']
    hits = measures['hits']

exact_matches = sum(1 for r in results if r["exact_match"])
exact_accuracy = exact_matches / len(results)
avg_latency = sum(r["latency_ms"] for r in results) / len(results)

# Отдельно для общих и с фамилиями
general_results = [r for r in results if r["phrase_id"] <= 7]
name_results = [r for r in results if r["phrase_id"] > 7]

gen_refs = [r["reference"] for r in general_results]
gen_hyps = [r["hypothesis"] if r["hypothesis"] else " " for r in general_results]
gen_wer = jiwer.wer(gen_refs, gen_hyps) if gen_refs else 0

name_refs = [r["reference"] for r in name_results]
name_hyps = [r["hypothesis"] if r["hypothesis"] else " " for r in name_results]
name_wer = jiwer.wer(name_refs, name_hyps) if name_refs else 0

print(f"\n{'='*60}")
print(f"  ASR BENCHMARK: GigaAM-CTC")
print(f"{'='*60}")
print(f"  Всего фраз:          {len(results)}")
print(f"  Точных совпадений:    {exact_matches}/{len(results)} ({exact_accuracy:.2%})")
print(f"{'='*60}")
print(f"  ОБЩИЙ WER:            {total_wer:.2%}")
print(f"  ОБЩИЙ CER:            {total_cer:.2%}")
print(f"{'='*60}")
print(f"  Substitutions:        {substitutions}")
print(f"  Deletions:            {deletions}")
print(f"  Insertions:           {insertions}")
print(f"  Hits:                 {hits}")
print(f"{'='*60}")
print(f"  WER (общие вопросы):  {gen_wer:.2%}")
print(f"  WER (с фамилиями):   {name_wer:.2%}")
print(f"{'='*60}")
print(f"  Avg Latency:          {avg_latency:.0f} ms")
print(f"{'='*60}")

# Сохраняем
report = {
    "model": "GigaAM-CTC",
    "total_phrases": len(results),
    "exact_match_accuracy": round(exact_accuracy, 4),
    "total_wer": round(total_wer, 4),
    "total_cer": round(total_cer, 4),
    "general_wer": round(gen_wer, 4),
    "names_wer": round(name_wer, 4),
    "substitutions": substitutions,
    "deletions": deletions,
    "insertions": insertions,
    "hits": hits,
    "avg_latency_ms": round(avg_latency, 1),
    "details": results,
}

with open("benchmark_asr_gigaam.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\nСохранено в benchmark_asr_gigaam.json")




