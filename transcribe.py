#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

VOCAB_MODEL = "gemini-3.8-flash"
STT_MODEL = "gemini-3.5-transcribe"

PRIMARY_CHUNK_SECONDS = 8 * 60
PRIMARY_OVERLAP_SECONDS = 4
LOW_WPM_THRESHOLD = 35.0
HIGH_WPM_THRESHOLD = 220.0
REPEAT_8GRAM_THRESHOLD = 8
MIN_WORDS_LONG_CHUNK = 20
HALF_OVERLAP_SECONDS = 2

LANGUAGE_CODES = ["ar-EG", "en-US"]
MAX_VOCAB_TERMS = 100
AUDIO_BITRATE = "48k"
AUDIO_SAMPLE_RATE = "16000"

MAX_API_ATTEMPTS = 4
RETRY_DELAYS_SECONDS = [5, 15, 30, 60]

# Hard network deadlines: no Gemini request can hang forever.
STT_TIMEOUT_SECONDS = 150
INLINE_AUDIO_LIMIT_BYTES = 18 * 1024 * 1024  # stay safely under the 20 MB request limit

SUPPORTED_EXTENSIONS = {
    ".mp4", ".m4a", ".mp3", ".wav", ".mov", ".mkv",
    ".aac", ".flac", ".ogg", ".webm", ".mpeg", ".mpg"
}

PROJECT_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = PROJECT_DIR / "transcripts"
ENV_FILE = PROJECT_DIR / ".env"


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


def load_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key

    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    os.environ["GEMINI_API_KEY"] = key
                    return key

    print("\nFirst run: paste your Gemini API key.")
    print("Get one from Google AI Studio: https://aistudio.google.com/apikey")
    print("The key is saved only in this project's local .env file.\n")
    key = getpass.getpass("GEMINI_API_KEY: ").strip()
    if not key:
        raise RuntimeError("No API key provided.")

    ENV_FILE.write_text(f"GEMINI_API_KEY={key}\n", encoding="utf-8")
    os.environ["GEMINI_API_KEY"] = key
    print("API key saved locally in .env.")
    return key


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
                ("Audio / Video", "*.mp4 *.m4a *.mp3 *.wav *.mov *.mkv *.aac *.flac *.ogg *.webm *.mpeg *.mpg"),
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
    except Exception as e:
        raise RuntimeError(
            "FFmpeg helper is unavailable. Run RUN_TRANSCRIBER.bat again so dependencies install."
        ) from e


def run_ffmpeg(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [get_ffmpeg(), "-hide_banner"] + args
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def get_duration_seconds(source: Path) -> float:
    p = run_ffmpeg(["-i", str(source)], capture=True)
    text = (p.stderr or "") + "\n" + (p.stdout or "")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        raise RuntimeError("Could not determine recording duration with FFmpeg.")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def extract_audio(source: Path, start: float, duration: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-y",
        "-ss", f"{max(0.0, start):.3f}",
        "-t", f"{max(0.1, duration):.3f}",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", AUDIO_SAMPLE_RATE,
        "-b:a", AUDIO_BITRATE,
        "-c:a", "libmp3lame",
        str(out_path),
    ]
    p = run_ffmpeg(args, capture=True)
    if p.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1000:
        raise RuntimeError(f"FFmpeg failed while creating {out_path.name}:\n{p.stderr[-2000:]}")


def extract_full_audio(source: Path, out_path: Path, duration: float) -> None:
    if out_path.exists() and out_path.stat().st_size > 1000:
        return
    print("Preparing compact speech audio...")
    extract_audio(source, 0, duration + 1, out_path)


def retryable_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = [
        "429", "resource_exhausted", "rate limit", "quota",
        "503", "500", "502", "504", "unavailable",
        "timeout", "timed out", "connection", "temporarily",
    ]
    return any(m in text for m in markers)


def call_with_retry(label: str, fn):
    last = None
    for attempt in range(MAX_API_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not retryable_error(exc) or attempt == MAX_API_ATTEMPTS - 1:
                break
            delay = RETRY_DELAYS_SECONDS[min(attempt, len(RETRY_DELAYS_SECONDS) - 1)]
            print(
                f"{label}: temporary API/network error "
                f"({type(exc).__name__}: {exc}). Retrying in {delay}s..."
            )
            time.sleep(delay)
    raise RuntimeError(
        f"{label} failed after retries.\n"
        f"Progress already saved. Run the same lecture again later to resume.\n"
        f"Last error: {last}"
    ) from last


def call_once(label: str, fn):
    """Single best-effort API call for optional enhancement stages."""
    try:
        return fn()
    except Exception as exc:
        raise RuntimeError(
            f"{label} failed: {type(exc).__name__}: {exc}"
        ) from exc


def upload_file(client, path: Path):
    """Fallback for unexpectedly large audio. Normal chunks are sent inline."""
    return call_with_retry(
        f"Upload {path.name}",
        lambda: client.files.upload(file=str(path)),
    )


def audio_input(client, path: Path) -> dict[str, str]:
    """
    Build Gemini audio input.

    Our 48 kbps 8-minute MP3 chunks are only a few MB, so they are sent
    inline. This avoids a separate Files API upload and the stalls we saw
    there. Files API is retained only as a safety fallback for >18 MB.
    """
    size = path.stat().st_size
    if size <= INLINE_AUDIO_LIMIT_BYTES:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "audio",
            "data": data,
            "mime_type": "audio/mp3",
        }

    print(
        f"    {path.name} is {size / (1024**2):.1f} MB; "
        "using Files API fallback."
    )
    uploaded = upload_file(client, path)
    return {
        "type": "audio",
        "uri": uploaded.uri,
        "mime_type": uploaded.mime_type,
    }


def normalize_vocab_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^```(?:text)?\s*", "", line, flags=re.I)
    line = line.replace("```", "").strip()
    line = re.sub(r"^\s*(?:[-*•]+|\d+[\.\):])\s*", "", line)
    return line.strip(" \t\"'`")


def parse_vocabulary(raw: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        term = normalize_vocab_line(line)
        if not term or not re.search(r"[A-Za-z]", term):
            continue
        if len(term) > 80 or len(term.split()) > 8:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= MAX_VOCAB_TERMS:
            break
    return terms


VOCAB_PROMPT = """Listen carefully to this university lecture recording.

The lecturer primarily speaks Egyptian Arabic and frequently code-switches into English technical terminology.

Your ONLY task is to identify technical English terms, acronyms, model names, dataset names, software/library names, researcher/proper names, and specialized English phrases that are ACTUALLY SPOKEN or clearly audible in this recording.

Rules:
- Do not summarize the lecture.
- Do not infer extra vocabulary merely because it belongs to the subject.
- Preserve the likely standard English spelling/capitalization.
- Prefer specific domain terms over ordinary English words.
- Include variants only when genuinely useful.
- Maximum 100 terms.
- Output exactly one term per line.
- No bullets, numbering, headings, explanations, or commentary.
"""



def merge_vocab(existing: list[str], new_terms: list[str], limit: int = MAX_VOCAB_TERMS) -> list[str]:
    """Merge vocabulary case-insensitively while preserving order."""
    merged: list[str] = []
    seen: set[str] = set()
    for term in list(new_terms) + list(existing):
        t = term.strip()
        if not t:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(t)
        if len(merged) >= limit:
            break
    return merged


def save_global_vocabulary(out_dir: Path, terms: list[str]) -> None:
    (out_dir / "auto_vocabulary.txt").write_text(
        "\n".join(terms) + ("\n" if terms else ""),
        encoding="utf-8"
    )



def discover_terms_from_transcript(
    client,
    transcript_text: str,
    chunk_index: int,
    out_dir: Path,
) -> list[str]:
    """
    Fast TEXT-only vocabulary discovery after transcription.

    This avoids sending the same audio to two different models. Any technical
    terms recovered from chunk N are carried forward as custom vocabulary for
    chunk N+1.
    """
    chunk_vocab_file = out_dir / f"vocab_chunk_{chunk_index:03d}.txt"

    if chunk_vocab_file.exists():
        saved = [
            x.strip()
            for x in chunk_vocab_file.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        if saved:
            return saved

    if not transcript_text.strip():
        return []

    prompt = """Extract technical English terms, acronyms, model names, dataset names,
software/library names, researcher/proper names, and specialized English phrases
that actually appear in the transcript below.

The transcript is from an Egyptian Arabic university lecture with English
code-switching.

Rules:
- Do not summarize.
- Preserve standard English spelling/capitalization where obvious.
- Do not invent subject-related terms that are absent.
- Maximum 40 terms.
- Output one term per line.
- No bullets, numbering, headings, or explanations.

TRANSCRIPT:
""" + transcript_text[:18000]

    def _request():
        return client.interactions.create(
            model=VOCAB_MODEL,
            input=prompt,
            timeout=30,
        )

    try:
        interaction = call_once(
            f"Text vocabulary extraction chunk {chunk_index:03d}",
            _request
        )
        raw = (interaction.output_text or "").strip()
        terms = parse_vocabulary(raw)[:40]
    except Exception as exc:
        print(f"    Vocabulary text extraction skipped: {exc}")
        terms = []

    chunk_vocab_file.write_text(
        "\n".join(terms) + ("\n" if terms else ""),
        encoding="utf-8"
    )
    return terms


def transcribe_audio(client, audio_path: Path, vocabulary: list[str], mode: str = "verbatim") -> str:
    """
    Transcribe one audio chunk.

    Gemini's official transcription examples use the Files API for audio.
    Upload once, then pass the returned URI to gemini-3.5-transcribe.
    """
    size_mb = audio_path.stat().st_size / (1024 ** 2)
    print(f"    Uploading {audio_path.name} ({size_mb:.1f} MB) for STT...")
    uploaded = upload_file(client, audio_path)
    print("    Audio uploaded; waiting for Gemini transcription...")

    tcfg: dict[str, Any] = {"language_codes": LANGUAGE_CODES}
    if vocabulary:
        tcfg["custom_vocabulary"] = vocabulary[:MAX_VOCAB_TERMS]
    tcfg["mode"] = "smart" if mode == "smart" else {"type": "verbatim"}

    def _request():
        return client.interactions.create(
            model=STT_MODEL,
            input=[{
                "type": "audio",
                "uri": uploaded.uri,
                "mime_type": uploaded.mime_type,
            }],
            generation_config={"transcription_config": tcfg},
            timeout=STT_TIMEOUT_SECONDS,
        )

    interaction = call_once(f"Transcribe {audio_path.name}", _request)
    return (interaction.output_text or "").strip()

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
        gram = tuple(words[i:i+n])
        counts[gram] = counts.get(gram, 0) + 1
        maximum = max(maximum, counts[gram])
    return maximum


def qc_metrics(text: str, duration_s: float) -> dict[str, Any]:
    words = word_count(text)
    minutes = max(duration_s / 60.0, 0.01)
    return {
        "words": words,
        "wpm": words / minutes,
        "repeat8": max_ngram_repeat(text, 8),
    }


def core_qc_reasons(metrics: dict[str, Any], duration_s: float) -> list[str]:
    reasons: list[str] = []
    if duration_s >= 60 and metrics["words"] < MIN_WORDS_LONG_CHUNK:
        reasons.append("almost-empty")
    if metrics["wpm"] > HIGH_WPM_THRESHOLD:
        reasons.append("implausibly-high-wpm")
    if metrics["repeat8"] >= REPEAT_8GRAM_THRESHOLD:
        reasons.append("repetition-loop")
    return reasons


def primary_qc_reasons(metrics: dict[str, Any], duration_s: float) -> list[str]:
    reasons = core_qc_reasons(metrics, duration_s)
    if duration_s >= 6 * 60 and metrics["wpm"] < LOW_WPM_THRESHOLD:
        reasons.append("low-wpm-one-time-check")
    return reasons


def save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def source_signature(source: Path) -> dict[str, Any]:
    st = source.stat()
    return {"name": source.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def build_primary_chunks(total_duration: float) -> list[dict[str, float]]:
    chunks = []
    start = 0.0
    idx = 1
    while start < total_duration - 0.25:
        end = min(start + PRIMARY_CHUNK_SECONDS, total_duration)
        chunks.append({"index": idx, "start": start, "end": end, "duration": end - start})
        if end >= total_duration:
            break
        start = end - PRIMARY_OVERLAP_SECONDS
        idx += 1
    return chunks


def transcribe_half_once(
    client, source: Path, vocabulary: list[str], work_dir: Path,
    chunk_index: int, label: str, start: float, end: float
) -> dict[str, Any]:
    duration = max(0.1, end - start)
    audio_path = work_dir / f"chunk_{chunk_index:03d}_{label}.mp3"
    extract_audio(source, start, duration, audio_path)

    text = transcribe_audio(client, audio_path, vocabulary, mode="verbatim")
    metrics = qc_metrics(text, duration)
    reasons = core_qc_reasons(metrics, duration)  # deliberately NO low-WPM check
    mode = "verbatim-4m"

    if reasons:
        print(f"      {label}: core QC flagged {', '.join(reasons)}; trying Smart once.")
        smart = transcribe_audio(client, audio_path, vocabulary, mode="smart")
        smart_metrics = qc_metrics(smart, duration)
        smart_reasons = core_qc_reasons(smart_metrics, duration)
        if len(smart_reasons) <= len(reasons):
            text, metrics, reasons, mode = smart, smart_metrics, smart_reasons, "smart-4m"

    return {
        "start": start, "end": end, "duration": duration, "mode": mode,
        "text": text, "metrics": metrics, "reasons": reasons,
        "status": "OK" if not reasons else "NEEDS_REVIEW",
    }


def transcribe_primary_chunk(
    client, source: Path, global_vocabulary: list[str],
    work_dir: Path, out_dir: Path, chunk: dict[str, float]
) -> tuple[dict[str, Any], list[str]]:
    idx = int(chunk["index"])
    start, end, duration = chunk["start"], chunk["end"], chunk["duration"]
    audio_path = work_dir / f"chunk_{idx:03d}.mp3"

    extract_audio(source, start, duration, audio_path)

    # Transcribe immediately using vocabulary accumulated from previous chunks.
    # No second audio-model pass is performed.
    vocabulary = list(global_vocabulary)[:MAX_VOCAB_TERMS]
    if vocabulary:
        print(f"    Using {len(vocabulary)} accumulated vocabulary terms.")
    else:
        print("    No accumulated vocabulary yet; transcribing directly.")

    try:
        text = transcribe_audio(client, audio_path, vocabulary, mode="verbatim")
    except Exception as exc:
        print(f"    Primary 8-minute STT failed/stalled: {exc}")
        print("    Falling back immediately to two ~4-minute transcription calls.")

        midpoint = start + duration / 2.0
        h1_start = start
        h1_end = min(end, midpoint + HALF_OVERLAP_SECONDS / 2.0)
        h2_start = max(start, midpoint - HALF_OVERLAP_SECONDS / 2.0)
        h2_end = end

        half1 = transcribe_half_once(
            client, source, vocabulary, work_dir, idx, "A",
            h1_start, h1_end
        )
        half2 = transcribe_half_once(
            client, source, vocabulary, work_dir, idx, "B",
            h2_start, h2_end
        )

        combined_text = (
            f"[{fmt_time(half1['start'])}–{fmt_time(half1['end'])} | {half1['mode']}]\n"
            f"{half1['text'].strip()}\n\n"
            f"[{fmt_time(half2['start'])}–{fmt_time(half2['end'])} | {half2['mode']}]\n"
            f"{half2['text'].strip()}"
        ).strip()

        chunk_vocabulary = discover_terms_from_transcript(
            client, combined_text, idx, out_dir
        )
        vocabulary = merge_vocab(
            global_vocabulary,
            chunk_vocabulary,
            MAX_VOCAB_TERMS
        )
        save_global_vocabulary(out_dir, vocabulary)

        combined_metrics = qc_metrics(combined_text, duration)
        status = (
            "OK"
            if half1["status"] == "OK" and half2["status"] == "OK"
            else "NEEDS_REVIEW"
        )

        return ({
            "index": idx,
            "start": start,
            "end": end,
            "duration": duration,
            "mode": "split-after-stt-failure",
            "text": combined_text,
            "metrics": combined_metrics,
            "reasons": ["primary-stt-failure"],
            "status": status,
            "split": True,
            "halves": [half1, half2],
            "chunk_vocabulary": chunk_vocabulary,
        }, vocabulary)

    # Learn terminology from the TEXT result for subsequent chunks.
    chunk_vocabulary = discover_terms_from_transcript(
        client, text, idx, out_dir
    )
    if chunk_vocabulary:
        print(
            "    Learned terms for next chunks: "
            + ", ".join(chunk_vocabulary[:10])
            + (" ..." if len(chunk_vocabulary) > 10 else "")
        )

    vocabulary = merge_vocab(
        global_vocabulary,
        chunk_vocabulary,
        MAX_VOCAB_TERMS
    )
    save_global_vocabulary(out_dir, vocabulary)
    metrics = qc_metrics(text, duration)
    reasons = primary_qc_reasons(metrics, duration)

    if not reasons:
        return ({
            "index": idx, "start": start, "end": end, "duration": duration,
            "mode": "verbatim", "text": text, "metrics": metrics,
            "reasons": [], "status": "OK", "split": False,
            "chunk_vocabulary": chunk_vocabulary,
        }, vocabulary)

    print(f"    QC flagged: {', '.join(reasons)}")
    print("    Retrying this chunk ONCE as two ~4-minute pieces.")

    midpoint = start + duration / 2.0
    h1_start, h1_end = start, min(end, midpoint + HALF_OVERLAP_SECONDS / 2.0)
    h2_start, h2_end = max(start, midpoint - HALF_OVERLAP_SECONDS / 2.0), end

    half1 = transcribe_half_once(client, source, vocabulary, work_dir, idx, "A", h1_start, h1_end)
    half2 = transcribe_half_once(client, source, vocabulary, work_dir, idx, "B", h2_start, h2_end)

    combined_text = (
        f"[{fmt_time(half1['start'])}–{fmt_time(half1['end'])} | {half1['mode']}]\n"
        f"{half1['text'].strip()}\n\n"
        f"[{fmt_time(half2['start'])}–{fmt_time(half2['end'])} | {half2['mode']}]\n"
        f"{half2['text'].strip()}"
    ).strip()

    combined_metrics = qc_metrics(combined_text, duration)
    status = "OK" if half1["status"] == "OK" and half2["status"] == "OK" else "NEEDS_REVIEW"

    return ({
        "index": idx, "start": start, "end": end, "duration": duration,
        "mode": "split-once", "text": combined_text, "metrics": combined_metrics,
        "reasons": reasons, "status": status, "split": True,
        "halves": [half1, half2],
        "chunk_vocabulary": chunk_vocabulary,
    }, vocabulary)


def write_outputs(out_dir: Path, state: dict[str, Any]) -> None:
    entries = sorted(state.get("chunks", {}).values(), key=lambda x: int(x["index"]))
    transcript_parts = []
    qc_lines = ["chunk\tstart\tend\tmode\twords\twpm\trepetition\tstatus\tnote"]

    for e in entries:
        start_s, end_s = fmt_time(e["start"]), fmt_time(e["end"])
        transcript_parts.append(
            f"===== {start_s}–{end_s} | {e['mode']} =====\n{e.get('text','').strip()}"
        )
        m = e["metrics"]
        note = ",".join(e.get("reasons", [])) if e.get("reasons") else "-"
        qc_lines.append(
            f"{int(e['index']):03d}\t{start_s}\t{end_s}\t{e['mode']}\t"
            f"{m['words']} words\t{m['wpm']:.1f} wpm\trepeat8={m['repeat8']}\t"
            f"{e['status']}\t{note}"
        )
        chunk_txt = out_dir / (
            f"chunk_{int(e['index']):03d}_{start_s.replace(':','-')}_{end_s.replace(':','-')}.txt"
        )
        chunk_txt.write_text(e.get("text", "").strip() + "\n", encoding="utf-8")

    (out_dir / "transcript.txt").write_text(
        "\n\n".join(transcript_parts).strip() + "\n", encoding="utf-8"
    )
    (out_dir / "qc_report.txt").write_text(
        "\n".join(qc_lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    console_utf8()
    parser = argparse.ArgumentParser(
        description="Automatic Egyptian Arabic + English university lecture transcription."
    )
    parser.add_argument("recording", nargs="?", help="Path to audio/video lecture recording")
    parser.add_argument("--force", action="store_true", help="Start this lecture over from scratch.")
    args = parser.parse_args()

    source = Path(args.recording.strip('"')) if args.recording else choose_file_gui()
    source = source.expanduser().resolve()

    if not source.exists() or not source.is_file():
        print(f"\nERROR: File not found:\n{source}")
        return 2
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"\nWARNING: Unusual extension {source.suffix}; FFmpeg will still try.")

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    out_dir = TRANSCRIPTS_DIR / safe_name(source.stem)
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_working"
    work_dir.mkdir(exist_ok=True)
    state_path = out_dir / "run_state.json"

    sig = source_signature(source)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("source") != sig:
            print("\nERROR: This output folder belongs to a different source file.")
            print("Rename the recording, delete the old transcript folder, or use --force.")
            return 3
    else:
        state = {
            "version": 1,
            "source": sig,
            "models": {"vocabulary": VOCAB_MODEL, "transcription": STT_MODEL},
            "chunks": {},
            "complete": False,
        }
        save_state(state_path, state)

    print("\n" + "=" * 68)
    print("LECTURE TRANSCRIBER")
    print("=" * 68)
    print(f"Recording : {source}")
    print(f"Output    : {out_dir}")
    print("GPU       : not required")
    print(f"STT       : {STT_MODEL}")
    print(f"Vocabulary: {VOCAB_MODEL}")

    api_key = load_key()
    try:
        from google import genai
    except ImportError:
        print("\nERROR: google-genai is not installed. Run RUN_TRANSCRIBER.bat again.")
        return 4
    client = genai.Client(api_key=api_key)

    duration = state.get("duration_seconds") or get_duration_seconds(source)
    state["duration_seconds"] = duration
    save_state(state_path, state)
    print(f"Duration  : {fmt_time(duration)}")

    if state.get("complete") and (out_dir / "transcript.txt").exists():
        print("\nThis lecture is already complete.")
        print(f"Transcript: {out_dir / 'transcript.txt'}")
        return 0

    chunks = build_primary_chunks(duration)
    global_vocabulary = list(state.get("vocabulary", []))
    vocab_file = out_dir / "auto_vocabulary.txt"
    if not global_vocabulary and vocab_file.exists():
        global_vocabulary = [
            x.strip()
            for x in vocab_file.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ][:MAX_VOCAB_TERMS]

    print(f"\nPrimary chunks: {len(chunks)} (~8 min each, 4 s overlap)")
    print("Vocabulary: learned from transcript text; STT uses Files API with 4-minute fallback on stalls.")
    print("Low-WPM QC: checked once on original chunks only (<35 WPM).")
    print("A flagged original chunk is split once into ~4-minute halves.\n")

    for chunk in chunks:
        idx = int(chunk["index"])
        key = str(idx)
        if key in state["chunks"]:
            e = state["chunks"][key]
            global_vocabulary = merge_vocab(
                global_vocabulary,
                e.get("chunk_vocabulary", []),
                MAX_VOCAB_TERMS
            )
            print(
                f"[{idx:02d}/{len(chunks):02d}] "
                f"{fmt_time(chunk['start'])}–{fmt_time(chunk['end'])}: "
                f"already saved ({e.get('status','OK')})"
            )
            continue

        print(
            f"[{idx:02d}/{len(chunks):02d}] "
            f"{fmt_time(chunk['start'])}–{fmt_time(chunk['end'])}: transcribing..."
        )
        entry, global_vocabulary = transcribe_primary_chunk(
            client, source, global_vocabulary, work_dir, out_dir, chunk
        )
        state["chunks"][key] = entry
        state["vocabulary"] = global_vocabulary
        save_global_vocabulary(out_dir, global_vocabulary)
        save_state(state_path, state)
        write_outputs(out_dir, state)

        m = entry["metrics"]
        print(
            f"    Saved: {m['words']} words, {m['wpm']:.1f} WPM, "
            f"repeat8={m['repeat8']}, {entry['status']}"
        )

    state["complete"] = len(state["chunks"]) == len(chunks)
    save_state(state_path, state)
    write_outputs(out_dir, state)

    if state["complete"]:
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass

    print("\n" + "=" * 68)
    print("DONE" if state["complete"] else "PARTIAL RUN SAVED")
    print("=" * 68)
    print(f"Transcript : {out_dir / 'transcript.txt'}")
    print(f"QC report  : {out_dir / 'qc_report.txt'}")
    print(f"Vocabulary : {out_dir / 'auto_vocabulary.txt'} (built progressively)")
    print("\nIf a connection/API quota interruption happens, run the SAME lecture again.")
    print("Completed chunks are skipped automatically.")
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
        print("\nSaved progress is kept. Rerun the same lecture to resume.")
        raise SystemExit(1)
