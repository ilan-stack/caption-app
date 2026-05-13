#!/usr/bin/env python3
"""Auto-caption pipeline: transcribe -> refine -> SRT -> burn-in."""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def transcribe(video_path, model_name="base", progress_cb=None):
    import whisper
    if progress_cb:
        progress_cb(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    if progress_cb:
        progress_cb("Transcribing audio (this may take a while)...")
    # word_timestamps=True returns per-word timing so we can snap segment
    # boundaries to actual speech instead of trusting Whisper's loose segment
    # bounds (which often start at 0.0 even when speech begins later).
    # condition_on_previous_text=False reduces hallucinated continuations.
    result = model.transcribe(
        str(video_path), verbose=False,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    segments = result["segments"]
    return _snap_segments_to_words(segments)


def _snap_segments_to_words(segments, *, pre_roll=0.05, post_roll=0.15,
                            min_dur=0.3):
    """Snap each segment's start/end to its actual first/last word timestamps.

    Whisper's segment-level start often extends well before the first spoken
    word (sometimes back to 0.0). Using `words[*].start` is much tighter.
    A tiny pre/post-roll keeps captions readable without leading the audio.
    """
    out = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            first = min(w["start"] for w in words)
            last = max(w["end"] for w in words)
            start = max(0.0, first - pre_roll)
            end = max(start + min_dur, last + post_roll)
        else:
            start, end = seg["start"], seg["end"]
        out.append({**seg, "start": start, "end": end})
    # Prevent overlaps if pre/post-rolls push segments into each other
    for i in range(1, len(out)):
        if out[i]["start"] < out[i - 1]["end"]:
            mid = (out[i - 1]["end"] + out[i]["start"]) / 2
            out[i - 1]["end"] = mid
            out[i]["start"] = mid
    return out


def refine_segments(segments, progress_cb=None):
    """Use Claude to clean grammar, punctuation, and remove fillers."""
    import anthropic
    client = anthropic.Anthropic()
    refined = []
    batch_size = 80
    total = len(segments)
    for i in range(0, total, batch_size):
        batch = segments[i:i + batch_size]
        if progress_cb:
            progress_cb(f"Refining segments {i + 1}-{i + len(batch)} of {total}...")
        lines = "\n".join(f"{j}: {s['text'].strip()}" for j, s in enumerate(batch))
        prompt = (
            "You are cleaning up auto-generated transcript segments for captioning. "
            "Fix grammar, punctuation, and capitalization. Remove filler words "
            "(um, uh, like, you know). Preserve meaning. Keep EXACTLY the same number "
            "of lines and the same numbering. Return ONLY the cleaned lines, one per "
            "input line, prefixed with the original index and a colon.\n\n" + lines
        )
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        out_text = resp.content[0].text.strip()
        out_lines = {}
        for line in out_text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            idx_str, txt = line.split(":", 1)
            try:
                out_lines[int(idx_str.strip())] = txt.strip()
            except ValueError:
                pass
        for j, s in enumerate(batch):
            new_text = out_lines.get(j, s["text"].strip())
            refined.append({**s, "text": new_text})
    return refined


def format_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap_text(text, max_chars):
    """Word-wrap text into lines no longer than max_chars."""
    words = (text or "").split()
    if not words:
        return ""
    lines, cur = [], words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return "\n".join(lines)


def apply_text_layout(segments, max_chars=42, max_lines=2):
    """Wrap each segment's text to `max_chars` per line. If wrapping produces
    more than `max_lines` lines, split into multiple sequential captions with
    time distributed proportionally to character count."""
    out = []
    for seg in segments:
        wrapped = wrap_text(seg["text"], max_chars)
        if not wrapped:
            continue
        lines = wrapped.split("\n")
        if len(lines) <= max_lines:
            out.append({**seg, "text": wrapped})
            continue
        duration = max(0.01, seg["end"] - seg["start"])
        total_chars = sum(len(l) for l in lines) or 1
        t = seg["start"]
        for i in range(0, len(lines), max_lines):
            chunk = lines[i:i + max_lines]
            chunk_chars = sum(len(l) for l in chunk) or 1
            chunk_dur = duration * (chunk_chars / total_chars)
            out.append({
                "start": t,
                "end": t + chunk_dur,
                "text": "\n".join(chunk),
            })
            t += chunk_dur
    return out


def write_srt(segments, srt_path):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, 1):
            text = s["text"].strip()
            f.write(f"{i}\n{format_ts(s['start'])} --> {format_ts(s['end'])}\n"
                    f"{text}\n\n")


def _format_ts_ass(seconds):
    """ASS uses H:MM:SS.cc (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def write_ass(segments, ass_path, video_w, video_h, pos_x=None, pos_y=None):
    """Write a minimal ASS file. libass parses `\\pos(x,y)` reliably from ASS
    (but strips it from SRT). Other styling is still applied via force_style."""
    prefix = ""
    if pos_x is not None and pos_y is not None:
        prefix = f"{{\\pos({int(pos_x)},{int(pos_y)})}}"
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_w}\n"
        f"PlayResY: {video_h}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,0,2,20,20,40,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for s in segments:
            text = s["text"].strip().replace("\n", "\\N")
            f.write(
                f"Dialogue: 0,{_format_ts_ass(s['start'])},"
                f"{_format_ts_ass(s['end'])},Default,,0,0,0,,{prefix}{text}\n"
            )


def get_video_dimensions(video_path):
    """Probe the input video for width and height using ffmpeg."""
    proc = subprocess.run(
        [_ffmpeg_bin(), "-i", str(video_path)],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
    )
    out = proc.stderr.decode("utf-8", "replace")
    # Look for "1920x1080" pattern
    import re
    m = re.search(r"(\d{2,5})x(\d{2,5})", out)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 640, 360  # fallback


def _ffmpeg_bin():
    """Prefer the bundled imageio-ffmpeg binary (has libass), fall back to system."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _rgb_to_ass(rgb_hex, alpha=0):
    """Convert '#RRGGBB' or 'RRGGBB' to ASS color &HAABBGGRR."""
    h = rgb_hex.lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b.upper()}{g.upper()}{r.upper()}"


_NUMPAD_TO_SSA = {1: 1, 2: 2, 3: 3, 4: 9, 5: 10, 6: 11, 7: 5, 8: 6, 9: 7}


def build_style(*, font_name="Arial", font_size=24,
                primary_color="#FFFFFF", outline_color="#000000",
                outline_width=2, alignment=2,
                margin_v=40, margin_l=20, margin_r=20,
                border_style=1, back_color="#000000", back_alpha=128,
                bold=False, italic=False,
                _mode=None, box_padding=8):
    """Compose an ASS Style string for libass force_style.

    `_mode` selects how the caption is rendered:
      None / "single" — regular single-pass: outline OR box (whichever border_style)
      "box_only"     — Pass 1 of two-pass: only the translucent box, text+outline invisible.
                       BorderStyle=4, Outline = box_padding.
      "text_only"    — Pass 2 of two-pass: only outlined text, no box.
                       BorderStyle=1, Outline = outline_width.
    """
    ssa_align = _NUMPAD_TO_SSA.get(int(alignment), 2)
    if _mode == "box_only":
        primary = "&HFFFFFFFF"   # alpha=FF -> fully transparent
        outline_col = "&HFFFFFFFF"
        back_col = _rgb_to_ass(back_color, 255 - back_alpha)
        border = 4
        outline = box_padding
    elif _mode == "text_only":
        primary = _rgb_to_ass(primary_color)
        outline_col = _rgb_to_ass(outline_color)
        back_col = "&HFF000000"  # transparent
        border = 1
        outline = outline_width
    else:
        primary = _rgb_to_ass(primary_color)
        outline_col = _rgb_to_ass(outline_color)
        back_col = _rgb_to_ass(back_color, 255 - back_alpha)
        border = border_style
        # libass overloads Outline as box-padding in BorderStyle=3/4
        outline = box_padding if border in (3, 4) else outline_width

    parts = [
        f"FontName={font_name}",
        f"FontSize={font_size}",
        f"PrimaryColour={primary}",
        f"OutlineColour={outline_col}",
        f"BackColour={back_col}",
        f"BorderStyle={border}",
        f"Outline={outline}",
        f"Shadow=0",
        f"Alignment={ssa_align}",
        f"MarginV={margin_v}",
        f"MarginL={margin_l}",
        f"MarginR={margin_r}",
        f"Bold={-1 if bold else 0}",
        f"Italic={-1 if italic else 0}",
    ]
    return ",".join(parts)


def burn_captions(video_path, srt_path, output_path, progress_cb=None,
                  **style_kwargs):
    border_style = style_kwargs.get("border_style", 1)
    outline_width = style_kwargs.get("outline_width", 0)
    # If user wants BOTH a translucent box AND a per-glyph outline, libass
    # can't do that in one pass — chain two subtitles filters: one renders
    # the box only, the next layers outlined text on top.
    two_pass = border_style in (3, 4) and outline_width > 0
    if two_pass:
        box_style = build_style(_mode="box_only", **style_kwargs)
        text_style = build_style(_mode="text_only", **style_kwargs)
        vf = (f"subtitles={srt_path}:force_style={box_style!r},"
              f"subtitles={srt_path}:force_style={text_style!r}")
        if progress_cb:
            progress_cb("Burning captions (two-pass: box + outlined text)...")
    else:
        style = build_style(**style_kwargs)
        vf = f"subtitles={srt_path}:force_style={style!r}"
        if progress_cb:
            progress_cb("Burning captions with FFmpeg...")

    cmd = [
        _ffmpeg_bin(), "-y", "-i", str(video_path),
        "-vf", vf,
        "-c:a", "copy",
        "-preset", "fast",
        str(output_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
        raise RuntimeError("ffmpeg failed:\n" + "\n".join(err))


def transcribe_only(video_path, model="base", refine=True, max_chars=42,
                    max_lines=2, progress_cb=None):
    """Run the transcribe + refine + layout stages and return segments JSON.
    Does NOT render the video. Used to support a two-step flow where the user
    edits/previews on a timeline before committing to the burn-in."""
    cb = progress_cb or (lambda m: print(m))
    segments = transcribe(video_path, model, cb)
    cb(f"Got {len(segments)} segments.")
    if not segments:
        raise RuntimeError("No speech detected in the audio.")
    if refine:
        if not os.getenv("ANTHROPIC_API_KEY"):
            cb("No ANTHROPIC_API_KEY set — skipping refinement.")
        else:
            segments = refine_segments(segments, cb)
    segments = apply_text_layout(segments, max_chars=max_chars, max_lines=max_lines)
    # Return plain dicts (no numpy types) so they round-trip through JSON
    return [{"start": float(s["start"]), "end": float(s["end"]),
             "text": s["text"].strip()} for s in segments]


ASS_REF_HEIGHT = 288  # libass's de-facto script-units reference (matches SRT path)


def write_ass_segments(segments, ass_path, video_w, video_h, default_pos=None):
    """Write ASS with optional per-segment `pos_x`/`pos_y` overrides.

    Uses `PlayResY = 288` (aspect-corrected `PlayResX`) so that FontSize, Outline,
    MarginV, etc. render at the SAME visual size as the SRT path — otherwise the
    moment a user drags a caption, everything snaps to a different scale. Caller
    passes positions in VIDEO PIXEL SPACE; we convert to script units here.

    Also force `\\an5` (middle-center anchor) on any dialogue with `\\pos`, so
    the drag point in the UI (which centers the overlay on the cursor) lines up
    pixel-perfectly with where the caption lands in the burned video.
    """
    play_res_y = ASS_REF_HEIGHT
    play_res_x = max(1, int(round(play_res_y * video_w / max(1, video_h))))
    px_to_script = play_res_y / max(1, video_h)

    def to_script(px, py):
        if px is None or py is None:
            return None, None
        try:
            px = float(px); py = float(py)
        except (TypeError, ValueError):
            return None, None
        if px < 0 or py < 0:
            return None, None
        return int(round(px * px_to_script)), int(round(py * px_to_script))

    dpx, dpy = to_script(*(default_pos or (None, None)))
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "ScaledBorderAndShadow: yes\n"
        # WrapStyle=2 → libass does NOT auto-wrap. Lines break only at
        # explicit `\N`. The server pre-wraps text upstream (apply_text_layout
        # / _rewrap_to_pct), so the rendered line breaks match exactly what
        # the in-browser preview shows. Without this, libass picks a
        # different greedy break point than CSS and the box width diverges.
        "WrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,0,2,20,20,40,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for s in segments:
            text = s["text"].strip().replace("\n", "\\N")
            spx, spy = to_script(s.get("pos_x"), s.get("pos_y"))
            sx = spx if spx is not None else dpx
            sy = spy if spy is not None else dpy
            prefix = ""
            if sx is not None and sy is not None:
                # \an5 = middle-center anchor in numpad encoding; pairs with the
                # drag UI which centers the overlay on the cursor.
                prefix = f"{{\\an5\\pos({sx},{sy})}}"
            f.write(
                f"Dialogue: 0,{_format_ts_ass(s['start'])},"
                f"{_format_ts_ass(s['end'])},Default,,0,0,0,,{prefix}{text}\n"
            )


def _rewrap_to_chars(segments, max_chars):
    """Word-wrap each segment's text to at most `max_chars` per line.

    Flattens any existing line breaks first so the wrap is recomputed from
    scratch each render — that way the user's CURRENT max_chars / wrap_pct
    sliders apply, even after /transcribe baked in a different layout.
    """
    if not max_chars or max_chars <= 0:
        return segments
    out = []
    for s in segments:
        flat = " ".join(s["text"].split())
        words = flat.split()
        lines, cur = [], ""
        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_chars:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        out.append({**s, "text": "\n".join(lines)})
    return out


def render_with_segments(video_path, segments, output_path, style_opts=None,
                         progress_cb=None):
    """Render burn-in given pre-computed segments + style. Honors per-segment
    `pos_x`/`pos_y` in each segment dict."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    cb = progress_cb or (lambda m: print(m))
    opts = dict(style_opts or {})
    default_pos_x = opts.pop("pos_x", None)
    default_pos_y = opts.pop("pos_y", None)
    max_chars = opts.pop("max_chars", 42) or 42
    opts.pop("max_lines", None)
    wrap_pct = opts.pop("wrap_pct", None)

    # Always re-wrap text on render so the line breaks match the user's
    # CURRENT max_chars + wrap_pct sliders (not whatever was baked at
    # /transcribe time). The in-browser preview uses the same wrap formula
    # — that's what keeps the box width identical between preview and burn.
    vw, vh = get_video_dimensions(video_path)
    font_size = opts.get("font_size", 24)
    char_budget = max_chars
    if wrap_pct and wrap_pct < 100:
        avg_char_px = max(1.0, font_size * (vh / 288.0) * 0.55)
        max_text_px = max(1.0, (wrap_pct / 100.0) * vw)
        char_budget = max(6, min(max_chars, int(max_text_px / avg_char_px)))
        cb(f"Wrap width = {wrap_pct}% ({char_budget} chars/line).")
    else:
        cb(f"Wrap width = max_chars ({char_budget} chars/line).")
    segments = _rewrap_to_chars(segments, char_budget)

    # Always use the ASS path so PlayResX/Y (aspect-corrected, PlayResY=288)
    # matches what the in-browser preview assumes. SRT goes through libass's
    # own default PlayRes which differs from ours, so the burn would wrap
    # text at a different width than the preview shows.
    vw, vh = get_video_dimensions(video_path)
    sub_path = Path(tempfile.mktemp(suffix=".ass"))
    write_ass_segments(segments, sub_path, vw, vh,
                       default_pos=(default_pos_x, default_pos_y))
    cb(f"ASS written ({vw}x{vh}, "
       f"{sum(1 for s in segments if s.get('pos_x') is not None)} per-seg positions)")

    burn_captions(video_path, sub_path, output_path, progress_cb=cb, **opts)
    try:
        sub_path.unlink()
    except OSError:
        pass
    return output_path


def run_pipeline(video_path, output_path=None, model="base", refine=True,
                 keep_srt=False, progress_cb=None, style_opts=None):
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.with_name(video_path.stem + "_captioned" + video_path.suffix)
    output_path = Path(output_path)
    cb = progress_cb or (lambda msg: print(msg))

    segments = transcribe(video_path, model, cb)
    cb(f"Got {len(segments)} segments.")

    if not segments:
        raise RuntimeError("No speech detected in the audio — nothing to caption.")

    if refine:
        if not os.getenv("ANTHROPIC_API_KEY"):
            cb("No ANTHROPIC_API_KEY set — skipping refinement.")
        else:
            segments = refine_segments(segments, cb)

    opts = dict(style_opts or {})
    max_chars = opts.pop("max_chars", 42)
    max_lines = opts.pop("max_lines", 2)
    pos_x = opts.pop("pos_x", None)
    pos_y = opts.pop("pos_y", None)
    segments = apply_text_layout(segments, max_chars=max_chars, max_lines=max_lines)
    cb(f"After layout: {len(segments)} captions, ≤{max_chars} chars × ≤{max_lines} lines.")

    use_ass = pos_x is not None and pos_y is not None
    if use_ass:
        vw, vh = get_video_dimensions(video_path)
        sub_path = (video_path.with_suffix(".ass") if keep_srt
                    else Path(tempfile.mktemp(suffix=".ass")))
        write_ass(segments, sub_path, vw, vh, pos_x=pos_x, pos_y=pos_y)
        cb(f"ASS written to {sub_path} (video {vw}x{vh}, pos=({pos_x},{pos_y}))")
    else:
        sub_path = (video_path.with_suffix(".srt") if keep_srt
                    else Path(tempfile.mktemp(suffix=".srt")))
        write_srt(segments, sub_path)
        cb(f"SRT written to {sub_path}")

    burn_captions(video_path, sub_path, output_path,
                  progress_cb=cb, **opts)
    cb(f"Done: {output_path}")

    if not keep_srt:
        try:
            sub_path.unlink()
        except OSError:
            pass
    return output_path


def main():
    p = argparse.ArgumentParser(description="Auto-caption a video.")
    p.add_argument("video")
    p.add_argument("--model", default="base",
                   choices=["tiny", "base", "small", "medium", "large"])
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--font-size", type=int, default=24)
    p.add_argument("--font-name", default="Arial")
    p.add_argument("--primary-color", default="#FFFFFF")
    p.add_argument("--outline-color", default="#000000")
    p.add_argument("--outline-width", type=int, default=2)
    p.add_argument("--alignment", type=int, default=2,
                   help="1-9 numpad layout: 1=BL 2=BC 3=BR 4=ML 5=MC 6=MR 7=TL 8=TC 9=TR")
    p.add_argument("--margin-v", type=int, default=40)
    p.add_argument("--margin-l", type=int, default=20)
    p.add_argument("--margin-r", type=int, default=20)
    p.add_argument("--keep-srt", action="store_true")
    p.add_argument("-o", "--output")
    args = p.parse_args()
    style_opts = dict(
        font_size=args.font_size, font_name=args.font_name,
        primary_color=args.primary_color, outline_color=args.outline_color,
        outline_width=args.outline_width, alignment=args.alignment,
        margin_v=args.margin_v, margin_l=args.margin_l, margin_r=args.margin_r,
    )
    run_pipeline(args.video, args.output, args.model, not args.no_refine,
                 args.keep_srt, style_opts=style_opts)


if __name__ == "__main__":
    main()
