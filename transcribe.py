#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


STATE_VERSION = 2
BACKEND = "faster-whisper"

DEFAULT_MODEL = "turbo"
DEFAULT_CHUNK_SECONDS = 5 * 60
DEFAULT_BEAM_SIZE = 5
DEFAULT_LANGUAGE = "auto"
DEFAULT_PROMPT = (
    "This is an Egyptian Arabic university lecture. The speaker frequently "
    "code-switches into English technical terminology, acronyms, library names, "
    "model names, equations, and proper nouns. Transcribe verbatim. Keep Arabic "
    "speech in Arabic script and keep English technical terms in Latin letters. "
    "Do not translate. Do not summarize."
)

LOW_WPM_THRESHOLD = 25.0
HIGH_WPM_THRESHOLD = 230.0
REPEAT_8GRAM_THRESHOLD = 6
MIN_WORDS_LONG_CHUNK = 12

SUPPORTED_EXTENSIONS = {
    ".mp4", ".m4a", ".mp3", ".wav", ".mov", ".mkv",
    ".aac", ".flac", ".ogg", ".webm", ".mpeg", ".mpg",
}

PROJECT_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = PROJECT_DIR / "transcripts"
MODEL_CACHE_DIR = PROJECT_DIR / "models"


def console_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name or "lecture"


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_time_label(label: str) -> str:
    return label.replace(":", "-")


def choose_file_gui() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filename = filedialog.askopenfilename(
            title="Choose a lecture recording",
            filetypes=[
                (
                    "Audio / Video",
                    "*.mp4 *.m4a *.mp3 *.wav *.mov *.mkv *.aac *.flac *.ogg *.webm *.mpeg *.mpg",
                ),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if filename:
            return Path(filename)
    except Exception:
        pass

    raw = input("Paste the full path to your lecture recording: ").strip().strip('"')
    return Path(raw)


def get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "FFmpeg helper is unavailable. Run RUN_TRANSCRIBER.bat so dependencies install."
        ) from exc


def run_ffmpeg(args: list[str], *, capture: bool = False, timeout: float = 120.0) -> subprocess.CompletedProcess:
    cmd = [get_ffmpeg(), "-hide_banner"] + args
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"FFmpeg timed out after {timeout:.0f}s.") from exc


def get_duration_seconds(source: Path) -> float:
    p = run_ffmpeg(["-i", str(source)], capture=True, timeout=45)
    text = (p.stderr or "") + "\n" + (p.stdout or "")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        raise RuntimeError("Could not determine recording duration with FFmpeg.")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def extract_audio(source: Path, start: float, duration: float, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 1000:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(90.0, duration * 3.0 + 30.0)
    args = [
        "-y",
        "-ss", f"{max(0.0, start):.3f}",
        "-t", f"{max(0.1, duration):.3f}",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    p = run_ffmpeg(args, capture=True, timeout=timeout)
    if p.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1000:
        tail = (p.stderr or "")[-2500:]
        raise RuntimeError(f"FFmpeg failed while creating {out_path.name}:\n{tail}")


def source_signature(source: Path) -> dict[str, Any]:
    st = source.stat()
    return {"name": source.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def build_chunks(total_duration: float, chunk_seconds: int) -> list[dict[str, float]]:
    chunks: list[dict[str, float]] = []
    start = 0.0
    index = 1
    while start < total_duration - 0.25:
        end = min(start + chunk_seconds, total_duration)
        chunks.append({"index": index, "start": start, "end": end, "duration": end - start})
        if end >= total_duration:
            break
        start = end
        index += 1
    return chunks


def save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def max_ngram_repeat(text: str, n: int = 8) -> int:
    cleaned = re.sub(r"[^\w\u0600-\u06FF]+", " ", text.lower(), flags=re.UNICODE)
    words = cleaned.split()
    if len(words) < n:
        return 1 if words else 0
    counts: dict[tuple[str, ...], int] = {}
    maximum = 1
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
        maximum = max(maximum, counts[gram])
    return maximum


def weighted_mean(items: Iterable[tuple[float, float]]) -> float | None:
    total_weight = 0.0
    total_value = 0.0
    for value, weight in items:
        if value is None:
            continue
        w = max(weight, 0.0)
        total_weight += w
        total_value += value * w
    if total_weight <= 0:
        return None
    return total_value / total_weight


def qc_metrics(text: str, duration_s: float, segments: list[dict[str, Any]]) -> dict[str, Any]:
    words = word_count(text)
    minutes = max(duration_s / 60.0, 0.01)
    speech_seconds = sum(max(0.0, s["end"] - s["start"]) for s in segments)
    avg_logprob = weighted_mean(
        (s.get("avg_logprob"), max(0.01, s["end"] - s["start"]))
        for s in segments
        if s.get("avg_logprob") is not None
    )
    no_speech_prob = weighted_mean(
        (s.get("no_speech_prob"), max(0.01, s["end"] - s["start"]))
        for s in segments
        if s.get("no_speech_prob") is not None
    )
    compression_ratio = max(
        [s.get("compression_ratio", 0.0) or 0.0 for s in segments] or [0.0]
    )
    return {
        "words": words,
        "wpm": words / minutes,
        "speech_seconds": speech_seconds,
        "speech_ratio": speech_seconds / max(duration_s, 0.01),
        "repeat8": max_ngram_repeat(text, 8),
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
        "max_compression_ratio": compression_ratio,
        "segments": len(segments),
    }


def qc_reasons(metrics: dict[str, Any], duration_s: float) -> list[str]:
    reasons: list[str] = []
    if duration_s >= 60 and metrics["words"] < MIN_WORDS_LONG_CHUNK:
        reasons.append("almost-empty")
    if duration_s >= 180 and metrics["wpm"] < LOW_WPM_THRESHOLD and metrics["speech_ratio"] > 0.18:
        reasons.append("low-wpm")
    if metrics["wpm"] > HIGH_WPM_THRESHOLD:
        reasons.append("implausibly-high-wpm")
    if metrics["repeat8"] >= REPEAT_8GRAM_THRESHOLD:
        reasons.append("repetition-loop")
    avg_logprob = metrics.get("avg_logprob")
    if avg_logprob is not None and avg_logprob < -1.25 and metrics["words"] > 10:
        reasons.append("low-confidence")
    if metrics.get("max_compression_ratio", 0.0) > 3.2 and metrics["words"] > 20:
        reasons.append("compression-loop-risk")
    return reasons


def segment_to_dict(segment: Any, chunk_start: float) -> dict[str, Any]:
    start = chunk_start + float(segment.start or 0.0)
    end = chunk_start + float(segment.end or segment.start or 0.0)
    return {
        "start": start,
        "end": end,
        "text": (segment.text or "").strip(),
        "avg_logprob": getattr(segment, "avg_logprob", None),
        "no_speech_prob": getattr(segment, "no_speech_prob", None),
        "compression_ratio": getattr(segment, "compression_ratio", None),
    }


def format_segment_text(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        lines.append(f"[{fmt_time(segment['start'])} - {fmt_time(segment['end'])}] {text}")
    return "\n".join(lines).strip()


def import_faster_whisper():
    try:
        from faster_whisper import WhisperModel

        return WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run RUN_TRANSCRIBER.bat to install dependencies."
        ) from exc


def model_candidates(device: str, compute_type: str) -> list[tuple[str, str]]:
    if device != "auto":
        compute = compute_type
        if compute == "auto":
            compute = "int8_float16" if device == "cuda" else "int8"
        return [(device, compute)]

    if compute_type != "auto":
        return [("cuda", compute_type), ("cpu", compute_type)]

    return [
        ("cuda", "int8_float16"),
        ("cuda", "float16"),
        ("cpu", "int8"),
    ]


def load_model(model_name: str, device: str, compute_type: str):
    WhisperModel = import_faster_whisper()
    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    errors: list[str] = []

    print("Loading Whisper model...")
    print(f"Model cache: {MODEL_CACHE_DIR}")
    for candidate_device, candidate_compute in model_candidates(device, compute_type):
        try:
            print(f"Trying {model_name} on {candidate_device} ({candidate_compute})...")
            model = WhisperModel(
                model_name,
                device=candidate_device,
                compute_type=candidate_compute,
                download_root=str(MODEL_CACHE_DIR),
            )
            return model, candidate_device, candidate_compute
        except Exception as exc:
            errors.append(f"{candidate_device}/{candidate_compute}: {type(exc).__name__}: {exc}")
            print(f"  Could not use {candidate_device}/{candidate_compute}: {exc}")

    joined = "\n".join(f"- {e}" for e in errors)
    raise RuntimeError(f"Could not load Whisper model with any device/compute setting:\n{joined}")


def transcribe_chunk(
    model: Any,
    audio_path: Path,
    chunk_start: float,
    *,
    language: str,
    beam_size: int,
    prompt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    language_arg = None if language == "auto" else language
    segments_iter, info = model.transcribe(
        str(audio_path),
        task="transcribe",
        language=language_arg,
        beam_size=beam_size,
        temperature=[0.0, 0.2, 0.4, 0.6],
        condition_on_previous_text=False,
        initial_prompt=prompt,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
        no_speech_threshold=0.65,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        word_timestamps=False,
    )
    segments = [segment_to_dict(segment, chunk_start) for segment in segments_iter]
    info_dict = {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "duration_after_vad": getattr(info, "duration_after_vad", None),
    }
    return segments, info_dict


def write_outputs(out_dir: Path, state: dict[str, Any]) -> None:
    entries = sorted(state.get("chunks", {}).values(), key=lambda x: int(x["index"]))
    transcript_parts: list[str] = []
    qc_lines = [
        "chunk\tstart\tend\tmodel\tdevice\tcompute\tlanguage\tsegments\twords\twpm\tspeech_ratio\tavg_logprob\tno_speech_prob\trepeat8\tstatus\tnote"
    ]
    jsonl_lines: list[str] = []

    for entry in entries:
        start_s = fmt_time(entry["start"])
        end_s = fmt_time(entry["end"])
        text = entry.get("text", "").strip()
        transcript_parts.append(
            f"===== {start_s} - {end_s} | {entry.get('backend', BACKEND)}:{entry.get('model')} =====\n{text}"
        )

        m = entry["metrics"]
        note = ",".join(entry.get("reasons", [])) if entry.get("reasons") else "-"
        avg_logprob = m.get("avg_logprob")
        no_speech_prob = m.get("no_speech_prob")
        avg_logprob_s = f"{avg_logprob:.3f}" if avg_logprob is not None else "-"
        no_speech_prob_s = f"{no_speech_prob:.3f}" if no_speech_prob is not None else "-"
        qc_lines.append(
            f"{int(entry['index']):03d}\t{start_s}\t{end_s}\t"
            f"{entry.get('model')}\t{entry.get('device')}\t{entry.get('compute_type')}\t"
            f"{entry.get('detected_language') or '-'}\t{m['segments']}\t{m['words']}\t"
            f"{m['wpm']:.1f}\t{m['speech_ratio']:.2f}\t"
            f"{avg_logprob_s}\t{no_speech_prob_s}\t{m['repeat8']}\t{entry['status']}\t{note}"
        )

        chunk_txt = out_dir / f"chunk_{int(entry['index']):03d}_{parse_time_label(start_s)}_{parse_time_label(end_s)}.txt"
        chunk_txt.write_text(text + ("\n" if text else ""), encoding="utf-8")

        for segment in entry.get("segments", []):
            jsonl_lines.append(json.dumps({
                "chunk": int(entry["index"]),
                "start": segment["start"],
                "end": segment["end"],
                "text": segment.get("text", ""),
                "avg_logprob": segment.get("avg_logprob"),
                "no_speech_prob": segment.get("no_speech_prob"),
                "compression_ratio": segment.get("compression_ratio"),
            }, ensure_ascii=False))

    transcript_text = "\n\n".join(part.strip() for part in transcript_parts if part.strip()).strip()
    (out_dir / "transcript.txt").write_text(transcript_text + ("\n" if transcript_text else ""), encoding="utf-8")
    (out_dir / "qc_report.txt").write_text("\n".join(qc_lines) + "\n", encoding="utf-8")
    (out_dir / "segments.jsonl").write_text("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""), encoding="utf-8")


def resolve_output_dir(source: Path, force: bool) -> Path:
    base = TRANSCRIPTS_DIR / safe_name(source.stem)
    if force:
        return base

    state_path = base / "run_state.json"
    if not state_path.exists():
        return base

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return base

    if state.get("backend") == BACKEND and state.get("version") == STATE_VERSION:
        return base

    return TRANSCRIPTS_DIR / f"{safe_name(source.stem)}_local_whisper"


def make_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": BACKEND,
        "model": args.model,
        "chunk_seconds": int(args.chunk_minutes * 60),
        "language": args.language,
        "beam_size": args.beam_size,
        "prompt": args.prompt,
    }


def load_or_create_state(
    state_path: Path,
    source: Path,
    settings: dict[str, Any],
    duration: float,
) -> dict[str, Any]:
    sig = source_signature(source)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("source") != sig:
            raise RuntimeError(
                "This output folder belongs to a different source file. "
                "Rename the recording, delete the old transcript folder, or use --force."
            )
        if state.get("backend") != BACKEND or state.get("version") != STATE_VERSION:
            raise RuntimeError(
                "This run_state.json belongs to the older Gemini pipeline. "
                "The local Whisper run should have been routed to a separate folder."
            )
        old_settings = state.get("settings", {})
        if old_settings != settings and state.get("chunks"):
            raise RuntimeError(
                "This lecture already has partial progress with different settings. "
                "Rerun without changing options, or use --force to start over."
            )
        state["duration_seconds"] = duration
        return state

    state = {
        "version": STATE_VERSION,
        "backend": BACKEND,
        "source": sig,
        "settings": settings,
        "duration_seconds": duration,
        "chunks": {},
        "complete": False,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_state(state_path, state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Egyptian Arabic + English lecture transcription using faster-whisper."
    )
    parser.add_argument("recording", nargs="?", help="Path to audio/video lecture recording")
    parser.add_argument("--force", action="store_true", help="Start this lecture over from scratch.")
    parser.add_argument("--model", default=os.getenv("WHISPER_MODEL", DEFAULT_MODEL),
                        help=f"faster-whisper model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=os.getenv("WHISPER_DEVICE", "auto"),
                        help="Inference device (default: auto).")
    parser.add_argument("--compute-type", default=os.getenv("WHISPER_COMPUTE_TYPE", "auto"),
                        help="Compute type, e.g. int8_float16, float16, int8 (default: auto).")
    parser.add_argument("--language", default=os.getenv("WHISPER_LANGUAGE", DEFAULT_LANGUAGE),
                        help="Language hint: auto or Whisper language code such as ar/en (default: auto).")
    parser.add_argument("--chunk-minutes", type=float, default=float(os.getenv("CHUNK_MINUTES", DEFAULT_CHUNK_SECONDS / 60)),
                        help="Checkpoint chunk size in minutes (default: 5).")
    parser.add_argument("--beam-size", type=int, default=int(os.getenv("WHISPER_BEAM_SIZE", DEFAULT_BEAM_SIZE)),
                        help="Whisper beam size (default: 5).")
    parser.add_argument("--prompt", default=os.getenv("WHISPER_PROMPT", DEFAULT_PROMPT),
                        help="Initial prompt used to preserve Arabic/English code-switching.")
    parser.add_argument("--keep-working", action="store_true", help="Keep extracted WAV chunks after completion.")
    return parser.parse_args()


def main() -> int:
    console_utf8()
    args = parse_args()

    if args.chunk_minutes <= 0:
        print("ERROR: --chunk-minutes must be positive.")
        return 2
    if args.beam_size <= 0:
        print("ERROR: --beam-size must be positive.")
        return 2

    source = Path(args.recording.strip('"')) if args.recording else choose_file_gui()
    source = source.expanduser().resolve()

    if not source.exists() or not source.is_file():
        print(f"\nERROR: File not found:\n{source}")
        return 2
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"\nWARNING: Unusual extension {source.suffix}; FFmpeg will still try.")

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    out_dir = resolve_output_dir(source, args.force)
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_working"
    work_dir.mkdir(exist_ok=True)
    state_path = out_dir / "run_state.json"

    print("\n" + "=" * 68)
    print("LECTURE TRANSCRIBER - LOCAL WHISPER")
    print("=" * 68)
    print(f"Recording : {source}")
    print(f"Output    : {out_dir}")
    print("Backend   : faster-whisper (local, no API key required)")
    print(f"Model     : {args.model}")
    print(f"Language  : {args.language}")

    duration = get_duration_seconds(source)
    print(f"Duration  : {fmt_time(duration)}")

    settings = make_settings(args)
    state = load_or_create_state(state_path, source, settings, duration)

    if state.get("complete") and (out_dir / "transcript.txt").exists():
        print("\nThis lecture is already complete.")
        print(f"Transcript: {out_dir / 'transcript.txt'}")
        return 0

    chunks = build_chunks(duration, int(args.chunk_minutes * 60))
    print(f"\nChunks    : {len(chunks)} ({args.chunk_minutes:g} min checkpoints)")
    print("Safety    : VAD on, previous-text conditioning off, progress saved after every chunk.")
    print("First run may download the Whisper model into the local models folder.\n")

    model, actual_device, actual_compute = load_model(args.model, args.device, args.compute_type)

    for chunk in chunks:
        idx = int(chunk["index"])
        key = str(idx)
        if key in state["chunks"]:
            entry = state["chunks"][key]
            print(
                f"[{idx:02d}/{len(chunks):02d}] "
                f"{fmt_time(chunk['start'])} - {fmt_time(chunk['end'])}: "
                f"already saved ({entry.get('status', 'OK')})"
            )
            continue

        print(
            f"[{idx:02d}/{len(chunks):02d}] "
            f"{fmt_time(chunk['start'])} - {fmt_time(chunk['end'])}: preparing audio..."
        )
        audio_path = work_dir / f"chunk_{idx:03d}.wav"
        extract_audio(source, chunk["start"], chunk["duration"], audio_path)

        print("    Transcribing...")
        started = time.monotonic()
        segments, info = transcribe_chunk(
            model,
            audio_path,
            chunk["start"],
            language=args.language,
            beam_size=args.beam_size,
            prompt=args.prompt,
        )
        elapsed = time.monotonic() - started
        text = format_segment_text(segments)
        metrics = qc_metrics(text, chunk["duration"], segments)
        reasons = qc_reasons(metrics, chunk["duration"])
        status = "OK" if not reasons else "NEEDS_REVIEW"

        entry = {
            "index": idx,
            "backend": BACKEND,
            "model": args.model,
            "device": actual_device,
            "compute_type": actual_compute,
            "start": chunk["start"],
            "end": chunk["end"],
            "duration": chunk["duration"],
            "elapsed_seconds": elapsed,
            "speed_factor": chunk["duration"] / max(elapsed, 0.01),
            "detected_language": info.get("language"),
            "language_probability": info.get("language_probability"),
            "text": text,
            "segments": segments,
            "metrics": metrics,
            "reasons": reasons,
            "status": status,
        }
        state["chunks"][key] = entry
        save_state(state_path, state)
        write_outputs(out_dir, state)

        print(
            f"    Saved: {metrics['words']} words, {metrics['wpm']:.1f} WPM, "
            f"{metrics['segments']} segments, {entry['speed_factor']:.1f}x realtime, {status}"
        )
        if reasons:
            print(f"    QC note: {', '.join(reasons)}")

    state["complete"] = len(state["chunks"]) == len(chunks)
    state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S") if state["complete"] else None
    save_state(state_path, state)
    write_outputs(out_dir, state)

    if state["complete"] and not args.keep_working:
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass

    print("\n" + "=" * 68)
    print("DONE" if state["complete"] else "PARTIAL RUN SAVED")
    print("=" * 68)
    print(f"Transcript : {out_dir / 'transcript.txt'}")
    print(f"QC report  : {out_dir / 'qc_report.txt'}")
    print(f"Segments   : {out_dir / 'segments.jsonl'}")
    print("\nIf the terminal closes, run the same recording again. Completed chunks are skipped.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n\nStopped. Completed chunks are checkpointed; rerun to resume.")
        raise SystemExit(130)
    except Exception as exc:
        print("\n" + "=" * 68)
        print("TRANSCRIPTION STOPPED")
        print("=" * 68)
        print(str(exc))
        print("\nSaved progress is kept. Rerun the same recording to resume.")
        raise SystemExit(1)
