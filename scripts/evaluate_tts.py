#!/usr/bin/env python3
"""Evaluate Piper TTS synthesis speed without running the assistant pipeline."""

from __future__ import annotations

import csv
import io
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.tts.synthesizer import Synthesizer  # noqa: E402

try:
    from src.tts.text_normalizer import normalize_for_tts
except ImportError:  # pragma: no cover - baseline compatibility fallback
    def normalize_for_tts(text: str) -> str:
        return text


TEST_PHRASES = [
    "Заведующий кафедрой — Блеканов Иван Станиславович.",
    "Кафедра технологии программирования была основана в 1989 году.",
    "Электронная почта: e.mitrofanov@spbu.ru.",
    "Электронная почта: s.popova@spbu.ru.",
    "Электронная почта кафедры: tp@spbu.ru.",
    "Попова Светлана Владимировна работает в кабинете 281.",
    "Митрофанов Евгений Павлович является преподавателем кафедры.",
    "Технологии программирования — это кафедра СПбГУ.",
    "В материалах кафедры нет информации по этому вопросу.",
    "Научные направления включают информационный поиск, анализ данных и вебометрику.",
]


def main() -> None:
    config = load_config()
    tts_config = config["tts"]
    synthesizer = Synthesizer(
        model_path=tts_config["model_path"],
        sample_rate=tts_config["sample_rate"],
    )
    synthesizer.load()

    # Warm-up is excluded from metrics.
    synthesizer.synthesize(normalize_for_tts("Проверка синтеза речи."))

    results = []
    for index, original_text in enumerate(TEST_PHRASES, start=1):
        normalized_text = normalize_for_tts(original_text)
        start = time.perf_counter()
        audio = synthesizer.synthesize(normalized_text)
        synthesis_time_sec = time.perf_counter() - start

        sample_rate = getattr(synthesizer, "sample_rate", tts_config["sample_rate"])
        audio_duration_sec = _audio_duration_sec(audio, sample_rate)
        rtf = synthesis_time_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0

        row = {
            "index": index,
            "original_text": original_text,
            "normalized_text": normalized_text,
            "changed_by_normalizer": normalized_text != original_text,
            "synthesis_time_sec": synthesis_time_sec,
            "audio_duration_sec": audio_duration_sec,
            "rtf": rtf,
            "sample_rate": sample_rate,
            "samples": _audio_size(audio),
        }
        results.append(row)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "tts_eval_results.csv"
    summary_path = results_dir / "tts_eval_summary.md"

    _write_csv(csv_path, results)
    _write_summary(summary_path, results)
    _print_table(results)


def _audio_duration_sec(audio: Any, sample_rate: int) -> float:
    if isinstance(audio, (bytes, bytearray)):
        wav_duration = _wav_bytes_duration(audio)
        if wav_duration is not None:
            return wav_duration
        return len(audio) / sample_rate if sample_rate > 0 else 0.0

    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if path.exists() and path.suffix.lower() == ".wav":
            return _wav_file_duration(path)

    try:
        return len(audio) / sample_rate if sample_rate > 0 else 0.0
    except TypeError:
        return 0.0


def _wav_bytes_duration(data: bytes | bytearray) -> float | None:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            return frames / sample_rate if sample_rate > 0 else 0.0
    except wave.Error:
        return None


def _wav_file_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        return frames / sample_rate if sample_rate > 0 else 0.0


def _audio_size(audio: Any) -> int:
    if isinstance(audio, (bytes, bytearray)):
        return len(audio)
    try:
        return len(audio)
    except TypeError:
        return 0


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "original_text",
        "normalized_text",
        "changed_by_normalizer",
        "synthesis_time_sec",
        "audio_duration_sec",
        "rtf",
        "sample_rate",
        "samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    **row,
                    "synthesis_time_sec": f"{row['synthesis_time_sec']:.6f}",
                    "audio_duration_sec": f"{row['audio_duration_sec']:.6f}",
                    "rtf": f"{row['rtf']:.6f}",
                }
            )


def _write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    synthesis_times = [row["synthesis_time_sec"] for row in results]
    audio_durations = [row["audio_duration_sec"] for row in results]
    rtfs = [row["rtf"] for row in results]
    changed = [row for row in results if row["changed_by_normalizer"]]

    lines = [
        "# TTS Evaluation Summary",
        "",
        f"- phrases: {len(results)}",
        f"- average synthesis_time_sec: {statistics.mean(synthesis_times):.6f}",
        f"- median synthesis_time_sec: {statistics.median(synthesis_times):.6f}",
        f"- min synthesis_time_sec: {min(synthesis_times):.6f}",
        f"- max synthesis_time_sec: {max(synthesis_times):.6f}",
        f"- average audio_duration_sec: {statistics.mean(audio_durations):.6f}",
        f"- average RTF: {statistics.mean(rtfs):.6f}",
        f"- median RTF: {statistics.median(rtfs):.6f}",
        f"- min RTF: {min(rtfs):.6f}",
        f"- max RTF: {max(rtfs):.6f}",
        f"- changed_by_normalizer_count: {len(changed)}",
        "",
        "## Normalizer Changed Examples",
        "",
    ]

    for row in changed[:3]:
        lines.extend(
            [
                f"### Example {row['index']}",
                f"- original: {row['original_text']}",
                f"- normalized: {row['normalized_text']}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def _print_table(results: list[dict[str, Any]]) -> None:
    print(
        f"{'index':>5}  {'changed':>7}  {'synthesis_time_sec':>18}  "
        f"{'audio_duration_sec':>18}  {'rtf':>8}"
    )
    for row in results:
        print(
            f"{row['index']:>5}  {str(row['changed_by_normalizer']):>7}  "
            f"{row['synthesis_time_sec']:>18.6f}  "
            f"{row['audio_duration_sec']:>18.6f}  {row['rtf']:>8.6f}"
        )


if __name__ == "__main__":
    main()
