from __future__ import annotations

import json
from pathlib import Path

from genie_core.audio import transcribe_audio
from genie_core.video.screenshot import extract_screenshots
from genie_core.video.detect import get_video_info
from genie_core.pdf.split import split_pdf_to_images
from genie_core.llm import LMStudioClient, extract_json, merge_structured
from genie_core.report import esc, html_page
from genie_core.text import format_time


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

Relevance levels:
- high: the segment explicitly discusses content visible on this page
- medium: the segment is likely about this page (topical overlap, adjacent context)
- low: the segment is only loosely related to this page

Output ONLY valid JSON."""


# Self-contained merge instructions (merge_structured sends this as the user
# prompt with the JSON array appended; no system prompt is used).
SYNTHESIS_PROMPT = """You are a meeting report generator. You will receive a JSON array whose items
are per-page meeting analyses (page, page_summary, discussion_summary, key_points,
questions_raised, decisions) and/or partially merged reports. Merge everything
into ONE final structured meeting report.

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
      "disputes": [{"positions": ["side A view", "side B view"], "resolution": "how resolved or unresolved"}]
    }
  ]
}

Rules:
- Group related pages into the same topic; keep the "pages" list accurate (page numbers as integers)
- Detect disputes/disagreements and list both positions
- Do NOT include transcript quotes or source references; they are re-attached programmatically later
- Keep the original language of the content
- Output ONLY valid JSON"""


# Fallback/limit constants for the transcript window sent to the vision model.
_NEARBY_WINDOW_SECONDS = 120.0
_NEARBY_CHAR_BUDGET = 2000
_MAX_TRANSCRIPT_CHARS = 6000

# Token budget per merge call during report synthesis (conservative: half of
# an 8K context).
_MERGE_BUDGET_TOKENS = 4096

# Number of audio quotes backfilled per topic.
_BACKFILL_TOP_N = 5

# Keys kept on each per-page analysis when fed to the synthesis merge
# (matched_segments are stripped and backfilled programmatically afterwards).
_PAGE_MERGE_KEYS = (
    "page", "page_summary", "discussion_summary", "key_points",
    "questions_raised", "decisions",
)


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

    Per-page analyses are saved incrementally under <output_dir>/page_analyses/;
    on rerun, existing page analyses are loaded and the vision call is skipped
    (resume support).

    Returns {"report": str, "markdown": str, "html": str, "pages": int, "topics": int}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = Path(pdf_path).stem
    pdf_file_name = Path(pdf_path).name

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

    analyses_dir = output_dir / "page_analyses"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    page_analyses = []
    for i, page in enumerate(pages):
        if progress_callback:
            progress_callback("matching", 0.25 + 0.4 * (i + 1) / len(pages))

        # Resume: load a previously saved analysis for this page if present
        page_file = analyses_dir / ("page_%03d.json" % page["page"])
        if page_file.exists():
            analysis = json.loads(page_file.read_text(encoding="utf-8"))
            page_analyses.append(analysis)
            continue

        # Send nearby segments (this page's estimated range +/- overlap)
        est_start = max(0, i * page_duration - page_duration * 0.3)
        est_end = min(total_duration, (i + 1) * page_duration + page_duration * 0.3)
        nearby = [s for s in segments if s["end"] >= est_start and s["start"] <= est_end]
        if not nearby:
            nearby = _fallback_window(segments, (est_start + est_end) / 2)

        transcript_text = _format_segments(nearby)
        if len(transcript_text) > _MAX_TRANSCRIPT_CHARS:
            transcript_text = transcript_text[:_MAX_TRANSCRIPT_CHARS] + "\n[... transcript truncated ...]"

        # Dynamic per-page part only; the full instructions (with relevance
        # levels and schema) go in the system prompt.
        prompt = (
            "This is page %d of '%s'.\n\n"
            "Transcript:\n%s"
        ) % (page["page"], pdf_name, transcript_text)

        analysis = _vision_and_parse(
            vision_llm,
            prompt=prompt,
            image_path=page["path"],
            what="page %d analysis" % page["page"],
        )
        analysis["page"] = page["page"]
        analysis["image_path"] = page["path"]

        # Save incrementally so an interrupted run can resume
        page_file.write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        page_analyses.append(analysis)

    # Save aggregated per-page analysis
    pages_analysis_path = output_dir / "page_analyses.json"
    pages_analysis_path.write_text(
        json.dumps(page_analyses, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Step 4: Synthesize into final report (hierarchical merge under a token
    # budget; transcript quotes stripped before merging and backfilled after)
    if progress_callback:
        progress_callback("synthesizing", 0.7)

    stripped = [_strip_matched_segments(pa) for pa in page_analyses]
    if len(stripped) == 1:
        report = _complete_and_parse(
            text_llm,
            prompt="%s\n\n%s" % (SYNTHESIS_PROMPT, json.dumps(stripped, ensure_ascii=False)),
            what="report synthesis",
        )
    else:
        report = merge_structured(
            stripped,
            text_llm,
            merge_prompt=SYNTHESIS_PROMPT,
            budget_tokens=_MERGE_BUDGET_TOKENS,
        )
    if not isinstance(report, dict):
        raise RuntimeError(
            "Report synthesis returned %s, expected a JSON object" % type(report).__name__)

    _backfill_sources(report, page_analyses, pdf_file_name, top_n=_BACKFILL_TOP_N)

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


def _fallback_window(segments: list[dict], center: float) -> list[dict]:
    """Fixed-budget window around the estimated time point, used when the
    estimated page range matched no segments. Never returns the full
    transcript: +/- _NEARBY_WINDOW_SECONDS around `center`, and if that is
    still empty, the segments closest to `center` up to a character budget."""
    lo = center - _NEARBY_WINDOW_SECONDS
    hi = center + _NEARBY_WINDOW_SECONDS
    window = [s for s in segments if s["end"] >= lo and s["start"] <= hi]
    if window:
        return window

    # Nothing in the time window: take the closest segments up to the budget
    by_distance = sorted(segments, key=lambda s: abs(float(s["start"]) - center))
    picked = []
    chars = 0
    for seg in by_distance:
        text_len = len(str(seg.get("text", "")))
        if picked and chars + text_len > _NEARBY_CHAR_BUDGET:
            break
        picked.append(seg)
        chars += text_len
    picked.sort(key=lambda s: float(s["start"]))
    return picked


def _format_segments(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        lines.append("[%s-%s] %s" % (
            format_time(seg["start"]), format_time(seg["end"]), seg["text"]))
    return "\n".join(lines)


def _vision_and_parse(llm, prompt: str, image_path: str, what: str) -> dict:
    """Vision call + JSON extraction; one retry at temperature=0, then raise."""
    raw = llm.vision(prompt=prompt, image_path=image_path,
                     system=PAGE_MATCH_PROMPT, temperature=0.2)
    try:
        result = extract_json(raw)
        if isinstance(result, dict):
            return result
    except ValueError:
        pass

    raw = llm.vision(prompt=prompt, image_path=image_path,
                     system=PAGE_MATCH_PROMPT, temperature=0)
    try:
        result = extract_json(raw)
    except ValueError as e:
        raise RuntimeError(
            "Vision model returned unparseable JSON for %s (after retry at temperature=0): %s"
            % (what, e))
    if not isinstance(result, dict):
        raise RuntimeError(
            "Vision model returned JSON %s for %s, expected an object"
            % (type(result).__name__, what))
    return result


def _complete_and_parse(llm, prompt: str, what: str, system: str = None) -> dict:
    """Text LLM call + JSON extraction; one retry at temperature=0, then raise."""
    raw = llm.complete(prompt=prompt, system=system, temperature=0.2)
    try:
        result = extract_json(raw)
        if isinstance(result, dict):
            return result
    except ValueError:
        pass

    raw = llm.complete(prompt=prompt, system=system, temperature=0)
    try:
        result = extract_json(raw)
    except ValueError as e:
        raise RuntimeError(
            "LLM returned unparseable JSON for %s (after retry at temperature=0): %s"
            % (what, e))
    if not isinstance(result, dict):
        raise RuntimeError(
            "LLM returned JSON %s for %s, expected an object" % (type(result).__name__, what))
    return result


def _strip_matched_segments(analysis: dict) -> dict:
    """Keep only the merge-relevant keys of a per-page analysis (drops
    matched_segments and image_path before synthesis)."""
    return {k: analysis[k] for k in _PAGE_MERGE_KEYS if k in analysis}


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_RELEVANCE_RANK = {"high": 0, "medium": 1, "low": 2}


def _backfill_sources(report: dict, page_analyses: list, pdf_file_name: str,
                      top_n: int = _BACKFILL_TOP_N):
    """Programmatically rebuild each topic's sources from its pages: slide
    references plus the top-N matched transcript quotes (by relevance) of
    those pages. No LLM involved, so timestamps cannot be hallucinated."""
    by_page = {}
    for pa in page_analyses:
        num = _as_int(pa.get("page"))
        if num is not None:
            by_page[num] = pa

    for topic in report.get("topics", []) or []:
        if not isinstance(topic, dict):
            continue
        sources = []
        candidates = []
        for p in topic.get("pages", []) or []:
            num = _as_int(p)
            if num is None:
                continue
            sources.append({"type": "slide", "pdf_file": pdf_file_name, "page": num})
            pa = by_page.get(num)
            if not pa:
                continue
            for seg in pa.get("matched_segments", []) or []:
                if isinstance(seg, dict) and str(seg.get("text", "")).strip():
                    candidates.append(seg)

        # Rank by relevance (high > medium > low), dedupe, keep top N in
        # chronological order
        candidates.sort(key=lambda s: _RELEVANCE_RANK.get(
            str(s.get("relevance", "")).lower(), 3))
        seen = set()
        picked = []
        for seg in candidates:
            key = (str(seg.get("start")), str(seg.get("text", "")))
            if key in seen:
                continue
            seen.add(key)
            picked.append(seg)
            if len(picked) >= top_n:
                break
        picked.sort(key=lambda s: _coerce_seconds(s.get("start"), 0.0))

        for seg in picked:
            sources.append({
                "type": "audio",
                "timestamp_start": seg.get("start"),
                "timestamp_end": seg.get("end"),
                "text": seg.get("text", ""),
            })

        topic["sources"] = sources
    return report


def _coerce_seconds(value, default=0.0) -> float:
    """Best-effort conversion of an LLM-provided timestamp to seconds."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if not text:
            return default
        if ":" in text:
            try:
                nums = [float(p) for p in text.split(":")]
            except ValueError:
                return default
            secs = 0.0
            for n in nums:
                secs = secs * 60 + n
            return secs
        try:
            return float(text)
        except ValueError:
            return default
    return default


def _safe_format_time(value) -> str:
    """format_time that never raises on LLM-provided garbage (None, etc.)."""
    try:
        return format_time(value)
    except ValueError:
        return format_time(_coerce_seconds(value, 0.0))


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
                    s = _safe_format_time(src.get("timestamp_start", 0))
                    e = _safe_format_time(src.get("timestamp_end", 0))
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
                s = _safe_format_time(seg.get("start", 0))
                e = _safe_format_time(seg.get("end", 0))
                rel = seg.get("relevance", "")
                lines.append("> [%s-%s] (%s) \"%s\"" % (s, e, rel, seg.get("text", "")))
            lines.append("")

    return "\n".join(lines)


_HTML_CSS = """
body{font-family:sans-serif;max-width:1000px;margin:0 auto;padding:20px;line-height:1.6}
h1{color:#1a1a2e} h2{color:#16213e;border-bottom:2px solid #e94560;padding-bottom:5px}
.time{color:#e94560;font-family:monospace;font-weight:bold}
.page-ref{color:#0f3460;font-family:monospace}
blockquote{border-left:3px solid #e94560;padding-left:10px;color:#555;margin:10px 0}
.topic{margin:25px 0;padding:20px;background:#f8f9fa;border-radius:8px;border-left:4px solid #3498db}
.dispute{background:#fff3cd;padding:10px;border-radius:5px;margin:10px 0}
.page-detail{margin:15px 0;padding:15px;background:#fff;border:1px solid #dee2e6;border-radius:5px}
.source{font-size:0.9em;color:#666}
""".strip()


def _generate_html(report: dict, page_analyses: list, pdf_name: str) -> str:
    """Generate an HTML report (all LLM-derived text escaped)."""
    lines = []
    title = report.get("title", "Meeting Report")

    lines.append("<h1>%s</h1>" % esc(title))
    summary = report.get("overall_summary", "")
    if summary:
        lines.append("<p><strong>Summary:</strong> %s</p>" % esc(summary))

    for i, topic in enumerate(report.get("topics", []), 1):
        lines.append('<div class="topic">')
        lines.append("<h2>%d. %s</h2>" % (i, esc(topic.get("title", ""))))

        topic_pages = topic.get("pages", [])
        if topic_pages:
            lines.append('<p class="page-ref">Slides: %s pages %s</p>' % (
                esc(pdf_name), esc(", ".join(str(p) for p in topic_pages))))

        lines.append("<p>%s</p>" % esc(topic.get("summary", "")))

        items = []
        for point in topic.get("key_points", []):
            items.append("<li>%s</li>" % esc(point))
        for d in topic.get("decisions", []):
            items.append("<li><strong>Decision:</strong> %s</li>" % esc(d))
        for a in topic.get("action_items", []):
            items.append("<li>TODO: %s</li>" % esc(a))
        if items:
            lines.append("<ul>")
            lines.extend(items)
            lines.append("</ul>")

        for dispute in topic.get("disputes", []):
            lines.append('<div class="dispute"><strong>Dispute:</strong><ul>')
            for pos in dispute.get("positions", []):
                lines.append("<li>%s</li>" % esc(pos))
            resolution = dispute.get("resolution", "")
            if resolution:
                lines.append("<li><em>Resolution: %s</em></li>" % esc(resolution))
            lines.append("</ul></div>")

        for src in topic.get("sources", []):
            if src.get("type") == "audio":
                s = _safe_format_time(src.get("timestamp_start", 0))
                e = _safe_format_time(src.get("timestamp_end", 0))
                lines.append('<blockquote><span class="time">[%s-%s]</span> "%s"</blockquote>' % (
                    esc(s), esc(e), esc(src.get("text", ""))))
            elif src.get("type") == "slide":
                lines.append('<blockquote class="page-ref">[%s p.%s]</blockquote>' % (
                    esc(src.get("pdf_file", pdf_name)), esc(src.get("page", "?"))))

        lines.append("</div>")

    lines.append("<hr><h2>Per-Page Details</h2>")
    for pa in page_analyses:
        page_num = pa.get("page", "?")
        lines.append('<div class="page-detail">')
        lines.append("<h3>Page %s</h3>" % esc(page_num))
        lines.append("<p><strong>Content:</strong> %s</p>" % esc(pa.get("page_summary", "")))
        lines.append("<p><strong>Discussion:</strong> %s</p>" % esc(pa.get("discussion_summary", "")))
        for seg in pa.get("matched_segments", []):
            s = _safe_format_time(seg.get("start", 0))
            e = _safe_format_time(seg.get("end", 0))
            lines.append('<blockquote class="source"><span class="time">[%s-%s]</span> (%s) "%s"</blockquote>' % (
                esc(s), esc(e), esc(seg.get("relevance", "")), esc(seg.get("text", ""))))
        lines.append("</div>")

    return html_page(title, "\n".join(lines), css=_HTML_CSS)
