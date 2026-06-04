# Audio Subtitle Overlay (Python)

Real-time subtitles from mic and/or system audio with a transparent overlay bar.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configure

List devices:

```bash
python app.py --list-devices
```

Edit the settings in config.json.

## Run

```bash
python app.py
```

## Notes

- System audio capture uses WASAPI loopback on Windows.
- For mic + system, keep the same capture_sample_rate for both devices.
- Optional online mode: set transcription.engine to openai and online.enable to true, then set OPENAI_API_KEY.
# Audio Subtitle Overlay (Python)

Real-time subtitles from mic and/or system audio with a transparent overlay bar.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configure

List devices:

```bash
python app.py --list-devices
```

Edit the settings in config.json.

## Run

```bash
python app.py
```

## Notes

- System audio capture uses WASAPI loopback on Windows.
- For mic + system, keep the same capture_sample_rate for both devices.
- Optional online mode: set transcription.engine to openai and online.enable to true, then set OPENAI_API_KEY.
