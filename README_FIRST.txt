LECTURE TRANSCRIBER - WINDOWS QUICK START
=========================================

WHAT IT DOES
------------
Choose one lecture recording (MP4, M4A, MP3, WAV, MOV, MKV, etc.).

The program automatically:
1. extracts clean 16 kHz mono audio chunks locally with FFmpeg;
2. transcribes with faster-whisper on your laptop;
3. uses VAD to ignore long silence and reduce classroom-noise hallucinations;
4. uses a fixed Egyptian Arabic + English code-switching prompt;
5. saves progress after every ~5-minute chunk;
6. resumes after interruption instead of starting over;
7. writes transcript.txt, qc_report.txt, segments.jsonl, chunk text files, and run_state.json.

No Gemini API key is required. The first transcription may download the selected
Whisper model into this project's local models folder. That folder is ignored by Git.

FIRST RUN
---------
1. Extract this folder to a permanent location, for example:
   D:\University\LectureTranscriber_Local

2. Double-click:
   RUN_TRANSCRIBER.bat

3. First launch creates a private .venv and installs required Python packages.

4. If Python is missing, install Python 3.11 or newer:
   https://www.python.org/downloads/windows/
   Enable "Add python.exe to PATH" during installation.

5. A file picker opens. Select the lecture recording.

EVERY LATER RUN
---------------
Double-click RUN_TRANSCRIBER.bat -> choose lecture -> leave it running.

You can also drag a recording directly onto RUN_TRANSCRIBER.bat.

OUTPUT
------
transcripts\
  <lecture filename>\
    transcript.txt
    qc_report.txt
    segments.jsonl
    chunk_001_...
    chunk_002_...
    run_state.json

RESUME
------
If the terminal closes or Windows sleeps:
run the SAME lecture file again.

Completed chunks are skipped automatically.

START OVER
----------
Delete that lecture folder under transcripts\

or run:
.venv\Scripts\python.exe transcribe.py "D:\path\lecture.mp4" --force

USEFUL OPTIONS
--------------
Default model:
.venv\Scripts\python.exe transcribe.py "D:\path\lecture.mp4"

Prefer Arabic as a language hint:
.venv\Scripts\python.exe transcribe.py "D:\path\lecture.mp4" --language ar

Force CPU if CUDA libraries are not installed correctly:
.venv\Scripts\python.exe transcribe.py "D:\path\lecture.mp4" --device cpu

Try a more accurate but slower model:
.venv\Scripts\python.exe transcribe.py "D:\path\lecture.mp4" --model large-v3

NOTES
-----
RUN_TRANSCRIBER.bat uses the local Whisper pipeline.

RESET_API_KEY.bat only removes an old .env file if one exists. The current
pipeline does not need API keys.
