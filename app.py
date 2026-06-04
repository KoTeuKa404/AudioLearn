import argparse
import logging
import queue
import threading

from audio_capture import AudioPipeline, format_device_list
from config import load_config
from overlay import SubtitleOverlay
from subtitle_buffer import SubtitleBuffer
from transcriber import TranscribeWorker, build_transcriber


def main() -> None:
    parser = argparse.ArgumentParser(description="Live subtitle overlay")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(format_device_list())
        return

    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stop_event = threading.Event()
    text_queue: queue.Queue[str] = queue.Queue()
    buffer = SubtitleBuffer(max_lines=config["ui"]["max_lines"])

    def on_text(text: str) -> None:
        if buffer.add(text):
            text_queue.put(buffer.render())

    pipeline = AudioPipeline(config, stop_event)
    pipeline.start()

    transcriber = build_transcriber(config)
    worker = TranscribeWorker(
        segment_queue=pipeline.segment_queue,
        text_callback=on_text,
        transcriber=transcriber,
        config=config,
        stop_event=stop_event,
    )
    worker.start()

    overlay = SubtitleOverlay(config["ui"], text_queue, stop_event)
    try:
        overlay.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        pipeline.stop()
        worker.join(timeout=2)


if __name__ == "__main__":
    main()
