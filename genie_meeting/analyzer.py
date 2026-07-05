from __future__ import annotations

import json
from pathlib import Path

from genie_core.audio import transcribe_audio
from genie_core.video.screenshot import extract_screenshots
from genie_core.video.detect import get_video_info
from genie_core.pdf.split import split_pdf_to_images
from genie_core.llm import LMStudioClient


PAGE_MATCH_PROMPT = """You are a meeting content analyzer. You have:
1. A PDF page image (from presentation slides)
2. A list of timestamped transcript segments from the meeting recording

Your task: determine which transcript segments are discussing THIS specific page.
Consider visual cues — if the speaker mentions content visible on this page
(keywords, data, topics), that segment belongs to this page.

Return strict JSON:
{
  "page_summary": "what this page shows",
  "matched_segments": [
    {"start": 0.0, "end": 0.0, "text": "...", "relevance": "high/medium/low"}
  ],
  "discussion_summary": "summary of what was discussed about this page",
  "key_points": ["point 1", "point 2"],
  "questions_raised": ["any questions or disputes mentioned"],
  "decisions": ["decisions made regarding this page content"]
}

Output ONLY valid JSON."""


SYNTHESIS_PROMPT = """You are a meeting report generator. Given per-page analysis results from a meeting,
produce a final structured meeting report.

Output strict JSON:
{
  "title": "meeting title",
  "overall_summary": "1-3 sentence summary",
  "topics": [
    {
      "title": "topic name",
      "pages": [1, 2],
      "summary": "what was discussed",
      "key_points": ["point"],
      "decisions": ["decision"],
      "action_items": ["action"],
      "disputes": [{"positions": ["side A view", "side B view"], "resolution": "how resolved or unresolved"}],
      "sources": [
        {"type": "audio", "timestamp_start": 0.0, "timestamp_end": 0.0, "text": "quote"},
        {"type": "slide", "pdf_file": "file.pdf", "page": 1}
      ]
    }
  ]
}

Rules:
- Group related pages into the same topic
- Every claim must have a source reference (audio timestamp OR pdf page)
- Detect disputes/disagreements and list both positions
- Output ONLY valid JSON"""


def analyze_meeting(
    pdf_path: str,
    video_path: str,
    output_dir: str,
    language: str = "zh",
    whisper_model: str = "medium",
    text_model: str = "qwen3.6-35b-a3b-mtp",
    vision_model: str = "qwen3-vl",
    lm_studio_url: str = "http://localhost:1234/v1",
    progress_callback=None,
) -> dict:
    """Analyze a meeting recording + PDF slides into structured notes.

    Returns {"report": str, "markdown": str, "html": str, "pages": int, "topics": int}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = Path(pdf_path).stem

    # Step 1: Transcribe audio
    if progress_callback:
        progress_callback("transcribing", 0)

    segments = transcribe_audio(video_path, language=language, model=whisper_model)

    transcript_path = output_dir / "transcript.json"
    transcript_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    # Step 2: Split PDF to images
    if progress_callback:
        progress_callback("splitting_pdf", 0.15)

    images_dir = output_dir / "pages"
    pages = split_pdf_to_images(pdf_path, str(images_dir))

    # Step 3: Match transcript to each page using vision model
    if progress_callback:
        progress_callback("matching", 0.25)

    vision_llm = LMStudioClient(base_url=lm_studio_url, model=vision_model)
    text_llm = LMStudioClient(base_url=lm_studio_url, model=text_model)

    # Estimate time range per page (evenly divided)
    total_duration = segments[-1]["end"] if segments else 0
    page_duration = total_duration / max(len(pages), 1)

    page_analyses = []
    for i, page in enumerate(pages):
        if progress_callback:
            progress_callback("matching", 0.25 + 0.4 * (i + 1) / len(pages))

        # Send nearby segments (this page's estimated range +/- overlap)
        est_start = max(0, i * page_duration - page_duration * 0.3)
        est_end = min(total_duration, (i + 1) * page_duration + page_duration * 0.3)
        nearby = [s for s in segments if s["end"] >= est_start and s["start"] <= est_end]
        if not nearby:
            nearby = segments

        transcript_text = _format_segments(nearby)

        prompt = (
            "You are a meeting content analyzer. Look at this slide image and the transcript below.\n"
            "Determine which transcript segments discuss THIS page's content.\n\n"
            "This is page %d of '%s'.\n\n"
            "Transcript:\n%s\n\n"
            "Return JSON with: page_summary, matched_segments (with start/end/text/relevance), "
            "discussion_summary, key_points, questions_raised, decisions.\n"
            "Output ONLY valid JSON."
        ) % (page["page"], pdf_name, transcript_text)

        raw = vision_llm.vision(
            prompt=prompt,
            image_path=page["path"],
            temperature=0.2,
        )
        analysis = _parse_json(raw)
        analysis["page"] = page["page"]
        analysis["image_path"] = page["path"]
        page_analyses.append(analysis)

    # Save per-page analysis
    pages_analysis_path = output_dir / "page_analyses.json"
    pages_analysis_path.write_text(
        json.dumps(page_analyses, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Step 4: Synthesize into final report
    if progress_callback:
        progress_callback("synthesizing", 0.7)

    synthesis_input = json.dumps({
        "pdf_file": Path(pdf_path).name,
        "total_pages": len(pages),
        "page_analyses": page_analyses,
    }, ensure_ascii=False)

    raw_report = text_llm.complete(
        prompt=f"Synthesize this per-page meeting analysis into a final report:\n\n{synthesis_input}",
        system=SYNTHESIS_PROMPT,
        temperature=0.2,
    )
    report = _parse_json(raw_report)

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Step 5: Generate markdown and HTML
    if progress_callback:
        progress_callback("generating_output", 0.85)

    md_content = _generate_markdown(report, page_analyses, pdf_name)
    md_path = output_dir / "report.md"
    md_path.write_text(md_content, encoding="utf-8")

    html_content = _generate_html(report, page_analyses, pdf_name)
    html_path = output_dir / "report.html"
    html_path.write_text(html_content, encoding="utf-8")

    if progress_callback:
        progress_callback("done", 1.0)

    return {
        "report": str(report_path),
        "markdown": str(md_path),
        "html": str(html_path),
        "pages": len(pages),
        "topics": len(report.get("topics", [])),
    }


def _format_segments(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        m1, s1 = divmod(int(seg["start"]), 60)
        m2, s2 = divmod(int(seg["end"]), 60)
        lines.append(f"[{m1:02d}:{s1:02d}-{m2:02d}:{s2:02d}] {seg['text']}")
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try to find JSON object in the text
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "parse_failed", "raw": text[:500]}


def _format_time(seconds) -> str:
    try:
        seconds = float(seconds)
    except (ValueError, TypeError):
        return str(seconds)
    m = int(seconds // 60)
    s = int(seconds % 60)
    return "%02d:%02d" % (m, s)


def _generate_markdown(report: dict, page_analyses: list, pdf_name: str) -> str:
    lines = []
    title = report.get("title", "Meeting Report")
    lines.append("# %s\n" % title)

    summary = report.get("overall_summary", "")
    if summary:
        lines.append("**Summary:** %s\n" % summary)

    lines.append("---\n")

    # Per-topic sections
    for i, topic in enumerate(report.get("topics", []), 1):
        lines.append("## %d. %s\n" % (i, topic.get("title", "Untitled")))

        topic_pages = topic.get("pages", [])
        if topic_pages:
            lines.append("**Related slides:** %s pages %s\n" % (
                pdf_name, ", ".join(str(p) for p in topic_pages)))

        lines.append(topic.get("summary", "") + "\n")

        for point in topic.get("key_points", []):
            lines.append("- %s" % point)
        if topic.get("key_points"):
            lines.append("")

        for d in topic.get("decisions", []):
            lines.append("- **Decision:** %s" % d)
        if topic.get("decisions"):
            lines.append("")

        for a in topic.get("action_items", []):
            lines.append("- [ ] %s" % a)
        if topic.get("action_items"):
            lines.append("")

        disputes = topic.get("disputes", [])
        if disputes:
            lines.append("### Disputes")
            for dispute in disputes:
                positions = dispute.get("positions", [])
                for pos in positions:
                    lines.append("- %s" % pos)
                resolution = dispute.get("resolution", "")
                if resolution:
                    lines.append("- **Resolution:** %s" % resolution)
            lines.append("")

        sources = topic.get("sources", [])
        if sources:
            lines.append("### Sources")
            for src in sources:
                if src.get("type") == "audio":
                    s = _format_time(src.get("timestamp_start", 0))
                    e = _format_time(src.get("timestamp_end", 0))
                    lines.append("> [%s-%s] \"%s\"" % (s, e, src.get("text", "")))
                elif src.get("type") == "slide":
                    lines.append("> [%s p.%s]" % (src.get("pdf_file", pdf_name), src.get("page", "?")))
            lines.append("")

    # Per-page appendix
    lines.append("---\n")
    lines.append("## Appendix: Per-Page Details\n")
    for pa in page_analyses:
        page_num = pa.get("page", "?")
        lines.append("### Page %s\n" % page_num)
        lines.append("**Content:** %s\n" % pa.get("page_summary", ""))
        lines.append("**Discussion:** %s\n" % pa.get("discussion_summary", ""))

        matched = pa.get("matched_segments", [])
        if matched:
            for seg in matched:
                s = _format_time(seg.get("start", 0))
                e = _format_time(seg.get("end", 0))
                rel = seg.get("relevance", "")
                lines.append("> [%s-%s] (%s) \"%s\"" % (s, e, rel, seg.get("text", "")))
            lines.append("")

    return "\n".join(lines)


def _generate_html(report: dict, page_analyses: list, pdf_name: str) -> str:
    lines = []
    title = report.get("title", "Meeting Report")
    lines.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    lines.append("<title>%s</title>" % title)
    lines.append("<style>")
    lines.append("body{font-family:sans-serif;max-width:1000px;margin:0 auto;padding:20px;line-height:1.6}")
    lines.append("h1{color:#1a1a2e} h2{color:#16213e;border-bottom:2px solid #e94560;padding-bottom:5px}")
    lines.append(".time{color:#e94560;font-family:monospace;font-weight:bold}")
    lines.append(".page-ref{color:#0f3460;font-family:monospace}")
    lines.append("blockquote{border-left:3px solid #e94560;padding-left:10px;color:#555;margin:10px 0}")
    lines.append(".topic{margin:25px 0;padding:20px;background:#f8f9fa;border-radius:8px;border-left:4px solid #3498db}")
    lines.append(".dispute{background:#fff3cd;padding:10px;border-radius:5px;margin:10px 0}")
    lines.append(".page-detail{margin:15px 0;padding:15px;background:#fff;border:1px solid #dee2e6;border-radius:5px}")
    lines.append(".source{font-size:0.9em;color:#666}")
    lines.append("</style></head><body>")

    lines.append("<h1>%s</h1>" % title)
    summary = report.get("overall_summary", "")
    if summary:
        lines.append("<p><strong>Summary:</strong> %s</p>" % summary)

    for i, topic in enumerate(report.get("topics", []), 1):
        lines.append('<div class="topic">')
        lines.append("<h2>%d. %s</h2>" % (i, topic.get("title", "")))

        topic_pages = topic.get("pages", [])
        if topic_pages:
            lines.append('<p class="page-ref">Slides: %s pages %s</p>' % (
                pdf_name, ", ".join(str(p) for p in topic_pages)))

        lines.append("<p>%s</p>" % topic.get("summary", ""))

        for point in topic.get("key_points", []):
            lines.append("<li>%s</li>" % point)

        for d in topic.get("decisions", []):
            lines.append('<li><strong>Decision:</strong> %s</li>' % d)

        for dispute in topic.get("disputes", []):
            lines.append('<div class="dispute"><strong>Dispute:</strong><ul>')
            for pos in dispute.get("positions", []):
                lines.append("<li>%s</li>" % pos)
            resolution = dispute.get("resolution", "")
            if resolution:
                lines.append("<li><em>Resolution: %s</em></li>" % resolution)
            lines.append("</ul></div>")

        for src in topic.get("sources", []):
            if src.get("type") == "audio":
                s = _format_time(src.get("timestamp_start", 0))
                e = _format_time(src.get("timestamp_end", 0))
                lines.append('<blockquote><span class="time">[%s-%s]</span> "%s"</blockquote>' % (
                    s, e, src.get("text", "")))
            elif src.get("type") == "slide":
                lines.append('<blockquote class="page-ref">[%s p.%s]</blockquote>' % (
                    src.get("pdf_file", pdf_name), src.get("page", "?")))

        lines.append("</div>")

    lines.append("<hr><h2>Per-Page Details</h2>")
    for pa in page_analyses:
        page_num = pa.get("page", "?")
        lines.append('<div class="page-detail">')
        lines.append("<h3>Page %s</h3>" % page_num)
        lines.append("<p><strong>Content:</strong> %s</p>" % pa.get("page_summary", ""))
        lines.append("<p><strong>Discussion:</strong> %s</p>" % pa.get("discussion_summary", ""))
        for seg in pa.get("matched_segments", []):
            s = _format_time(seg.get("start", 0))
            e = _format_time(seg.get("end", 0))
            lines.append('<blockquote class="source"><span class="time">[%s-%s]</span> (%s) "%s"</blockquote>' % (
                s, e, seg.get("relevance", ""), seg.get("text", "")))
        lines.append("</div>")

    lines.append("</body></html>")
    return "\n".join(lines)
