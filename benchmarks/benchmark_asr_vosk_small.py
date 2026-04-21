import json
import time
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
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
    "почта Поповы Светланы Владимировны",
    "где найти Попову Светлану Владимировну",
    "как связаться с Митрофановой Ольгой Александровой",
    "почта Митрофановы Ольги Александровны",
    "где найти Митрофанову Ольгу Александровну",
    "почта заведующего кафедрой",
    "как связаться с заведующим",
]

RECORD_SECONDS = 4
SAMPLE_RATE = 16000

print("Загрузка модели Vosk...")
model = Model("data/models/vosk-model-small-ru-0.22")
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
        dtype="int16"
    )
    sd.wait()

    # Распознавание
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.AcceptWaveform(audio.tobytes())
    result = json.loads(rec.Result())
    recognized = result.get("text", "").strip()

    # Сравнение
    ref = phrase.lower().strip()
    hyp = recognized.lower().strip()

    word_wer = jiwer.wer(ref, hyp) if hyp else 1.0
    char_cer = jiwer.cer(ref, hyp) if hyp else 1.0

    match = "ВЕРНО" if ref == hyp else "ОШИБКА"

    print(f"  Эталон:    \"{ref}\"")
    print(f"  Распознал: \"{hyp}\"")
    print(f"  WER: {word_wer:.2%}  CER: {char_cer:.2%}  [{match}]")

    results.append({
        "phrase_id": i + 1,
        "reference": ref,
        "hypothesis": hyp,
        "exact_match": ref == hyp,
        "wer": round(word_wer, 4),
        "cer": round(char_cer, 4),
    })

# ============================================
# Итоговые метрики
# ============================================
all_refs = [r["reference"] for r in results]
all_hyps = [r["hypothesis"] for r in results]

# Заменяем пустые гипотезы на пробел чтобы jiwer не упал
all_hyps_safe = [h if h else " " for h in all_hyps]

total_wer = jiwer.wer(all_refs, all_hyps_safe)
total_cer = jiwer.cer(all_refs, all_hyps_safe)

# В новых версиях jiwer (3.x) вместо compute_measures используется process_words
try:
    # Новая версия jiwer 3.x
    output = jiwer.process_words(all_refs, all_hyps_safe)
    substitutions = output.substitutions
    deletions = output.deletions
    insertions = output.insertions
    hits = output.hits
    mer = substitutions + deletions + insertions
    total_words = hits + substitutions + deletions
    mer_rate = mer / total_words if total_words > 0 else 0
    wil = 1 - (hits / total_words) * (hits / (hits + insertions)) if total_words > 0 and (hits + insertions) > 0 else 0
    wip = 1 - wil
except AttributeError:
    # Старая версия jiwer 2.x
    measures = jiwer.compute_measures(all_refs, all_hyps_safe)
    substitutions = measures['substitutions']
    deletions = measures['deletions']
    insertions = measures['insertions']
    hits = measures['hits']
    mer_rate = measures['mer']
    wil = measures['wil']
    wip = measures['wip']

exact_matches = sum(1 for r in results if r["exact_match"])
exact_accuracy = exact_matches / len(results)

# Отдельно считаем для фраз с фамилиями и без
general_results = [r for r in results if r["phrase_id"] <= 7]
name_results = [r for r in results if r["phrase_id"] > 7]

gen_refs = [r["reference"] for r in general_results]
gen_hyps = [r["hypothesis"] if r["hypothesis"] else " " for r in general_results]
gen_wer = jiwer.wer(gen_refs, gen_hyps) if gen_refs else 0

name_refs = [r["reference"] for r in name_results]
name_hyps = [r["hypothesis"] if r["hypothesis"] else " " for r in name_results]
name_wer = jiwer.wer(name_refs, name_hyps) if name_refs else 0

print(f"\n{'='*60}")
print(f"  ASR BENCHMARK: vosk-model-small-ru")
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

# Сохраняем
report = {
    "model": "vosk-model-small-ru-0.22",
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
    "details": results,
}

with open("benchmark_asr_vosk_small.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\nСохранено в benchmark_asr_vosk_small.json")
