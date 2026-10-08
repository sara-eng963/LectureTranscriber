LECTURE TRANSCRIBER - WINDOWS QUICK START
==========================================

WHAT IT DOES
------------
Choose one lecture recording (MP4, M4A, MP3, WAV, MOV, MKV, etc.).

The program automatically:
1. extracts compact speech audio locally;
2. uses Gemini 3.8 Flash to discover technical English vocabulary from the lecture;
3. uses Gemini 3.5 Transcribe for Egyptian Arabic + English code-switching;
4. transcribes in ~8-minute chunks;
5. checks repetition, absurd output, and missing-text symptoms;
6. checks low WPM ONLY on the original ~8-minute chunk;
7. if an original chunk is under 35 WPM, splits it ONCE into ~4-minute halves;
8. does NOT apply low-WPM splitting again to those 4-minute halves;
9. saves progress after every chunk;
10. resumes after disconnects/API interruptions instead of starting over.

Your RTX 3050 is not used for model inference. Gemini runs remotely.
Your laptop handles FFmpeg, files, QC and checkpointing.

FIRST RUN
---------
1. Extract this ZIP to a permanent folder, for example:
   D:\University\LectureTranscriber_Local

2. Double-click:
   RUN_TRANSCRIBER.bat

3. First launch creates a private .venv and installs required Python packages.

4. If Python is missing, install Python 3.11 or newer:
   https://www.python.org/downloads/windows/
   Enable "Add python.exe to PATH" during installation.

5. The first run asks for a Gemini API key.
   Get one from:
   https://aistudio.google.com/apikey

   Paste it into the terminal.
   It is saved only in this folder as:
   .env

6. A file picker opens. Select the lecture recording.

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
    auto_vocabulary.txt
    auto_vocabulary_raw.txt
    chunk_001_...
    chunk_002_...
    run_state.json

RESUME
------
If Wi-Fi, the API, or the terminal interrupts:
run the SAME lecture file again.

Completed chunks are skipped automatically.

START OVER
----------
Delete that lecture folder under transcripts\

or run:
.venv\Scripts\python.exe transcribe.py "D:\path\lecture.mp4" --force

API KEY
-------
Double-click RESET_API_KEY.bat to remove the locally saved API key.

FREE-TIER NOTE
--------------
The current Gemini free tier lists Gemini 3.8 Flash and Gemini 3.5 Transcribe
input/output as free of charge, subject to Google project/account rate limits.
Free-tier data may be used by Google to improve its products.

DO NOT SHARE
------------
.env

It contains your Gemini API key.
