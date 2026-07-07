from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyzer import analyze_meeting


def main():
    parser = argparse.ArgumentParser(description="Analyze meeting recording + PDF into structured report")
    parser.add_argument("pdf", help="Path to PDF slides")
    parser.add_argument("video", help="Path to meeting recording (video/audio)")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--language", default="zh", help="Whisper language (default: zh)")
    parser.add_argument("--whisper-model", default="medium", help="Whisper model size")
    parser.add_argument("--text-model", default=None, help="LLM for text synthesis (default: auto-pick)")
    parser.add_argument("--vision-model", default=None, help="Vision model for page parsing (default: auto-pick)")
    parser.add_argument("--url", default="http://localhost:1234/v1", help="LM Studio API URL")

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    video_path = Path(args.video)

    if not pdf_path.exists():
        print("Error: PDF not found: %s" % pdf_path, file=sys.stderr)
        sys.exit(1)
    if not video_path.exists():
        print("Error: Video not found: %s" % video_path, file=sys.stderr)
        sys.exit(1)

    # Default output next to the input PDF (consistent with the other tools)
    output_dir = args.output or str(pdf_path.with_suffix("")) + "_meeting_report"

    def on_progress(stage, pct):
        stages = {
            "transcribing": "Transcribing audio",
            "splitting_pdf": "Splitting PDF",
            "matching": "Matching transcript to pages",
            "synthesizing": "Synthesizing report",
            "generating_output": "Generating output",
            "done": "Done",
        }
        label = stages.get(stage, stage)
        print("\r[%.0f%%] %s..." % (pct * 100, label), end="", flush=True)

    print("Processing: %s + %s" % (pdf_path, video_path))
    try:
        result = analyze_meeting(
            str(pdf_path),
            str(video_path),
            output_dir,
            language=args.language,
            whisper_model=args.whisper_model,
            text_model=args.text_model,
            vision_model=args.vision_model,
            lm_studio_url=args.url,
            progress_callback=on_progress,
        )
    except (RuntimeError, ValueError) as e:
        print("\nError: %s" % e, file=sys.stderr)
        sys.exit(1)
    print("\nDone! %d pages, %d topics" % (result["pages"], result["topics"]))
    print("  Report:   %s" % result["report"])
    print("  Markdown: %s" % result["markdown"])
    print("  HTML:     %s" % result["html"])


if __name__ == "__main__":
    main()
