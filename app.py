"""FastAPI web app: upload a video, get it back captioned."""
import asyncio
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import caption_pipeline as cp

app = FastAPI()

WORK_DIR = Path(os.environ.get("CAPTION_APP_DIR", "/tmp/caption-app"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

JOBS: dict = {}


def _job_record():
    return {"status": "queued", "log": [], "output": None, "error": None}


async def _run_job(job_id: str, src: Path, model: str, refine: bool,
                   style_opts: dict):
    job = JOBS[job_id]

    def log(msg):
        print(f"[{job_id[:8]}] {msg}")
        job["log"].append(msg)

    job["status"] = "running"
    log(f"Starting job: model={model}, refine={refine}")
    log(f"Style: {style_opts}")
    try:
        out_path = src.with_name(src.stem + "_captioned" + src.suffix)
        await asyncio.to_thread(
            cp.run_pipeline,
            src, out_path, model, refine, False, log, style_opts,
        )
        job["output"] = str(out_path)
        job["status"] = "done"
        log("Job complete.")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        log(f"ERROR: {e}")


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    model: str = Form("base"),
    refine: bool = Form(True),
    font_size: int = Form(24),
    font_name: str = Form("Arial"),
    primary_color: str = Form("#FFFFFF"),
    outline_color: str = Form("#000000"),
    outline_width: int = Form(2),
    alignment: int = Form(2),
    margin_v: int = Form(40),
    margin_l: int = Form(20),
    margin_r: int = Form(20),
    border_style: int = Form(1),
    back_color: str = Form("#000000"),
    back_alpha: int = Form(128),
    bold: bool = Form(False),
    italic: bool = Form(False),
    max_chars: int = Form(42),
    max_lines: int = Form(2),
    pos_x: int = Form(-1),  # -1 means "no custom position"
    pos_y: int = Form(-1),
    box_padding: int = Form(8),
):
    if not file.filename:
        raise HTTPException(400, "No filename")
    job_id = uuid.uuid4().hex
    safe_name = Path(file.filename).name
    src = WORK_DIR / f"{job_id}_{safe_name}"
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    style_opts = dict(
        font_size=font_size, font_name=font_name,
        primary_color=primary_color, outline_color=outline_color,
        outline_width=outline_width, alignment=alignment,
        margin_v=margin_v, margin_l=margin_l, margin_r=margin_r,
        border_style=border_style, back_color=back_color, back_alpha=back_alpha,
        bold=bold, italic=italic, box_padding=box_padding,
        max_chars=max_chars, max_lines=max_lines,
    )
    if pos_x >= 0 and pos_y >= 0:
        style_opts["pos_x"] = pos_x
        style_opts["pos_y"] = pos_y
    JOBS[job_id] = _job_record()
    asyncio.create_task(_run_job(job_id, src, model, refine, style_opts))
    return {"job_id": job_id}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("base"),
    refine: bool = Form(True),
    max_chars: int = Form(42),
    max_lines: int = Form(2),
):
    """Upload + transcribe in one step. Returns segments + a job_id the
    client can later POST to /render with style options and per-segment
    position overrides."""
    job_id = uuid.uuid4().hex
    safe_name = Path(file.filename or "video.mp4").name
    src = WORK_DIR / f"{job_id}_{safe_name}"
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    job = _job_record()
    job["src"] = str(src)
    job["status"] = "transcribing"
    JOBS[job_id] = job

    def log(msg):
        print(f"[{job_id[:8]}] {msg}")
        job["log"].append(msg)

    try:
        segments = await asyncio.to_thread(
            cp.transcribe_only, src,
            model, refine, max_chars, max_lines, log,
        )
        job["status"] = "transcribed"
        job["segments"] = segments
        # Also probe video duration for the timeline (best-effort)
        vw, vh = await asyncio.to_thread(cp.get_video_dimensions, src)
        return {
            "job_id": job_id,
            "segments": segments,
            "video_url": f"/video/{job_id}",
            "video_w": vw, "video_h": vh,
        }
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        log(f"ERROR: {e}")
        raise HTTPException(500, str(e))


@app.get("/video/{job_id}")
def get_video(job_id: str):
    """Stream the original uploaded video back so the client can play it."""
    job = JOBS.get(job_id)
    if not job or "src" not in job:
        raise HTTPException(404, "Unknown job")
    return FileResponse(job["src"])


@app.post("/render")
async def render(payload: dict = Body(...)):
    """Render the final captioned video from edited segments + style opts.
    Body: {job_id, segments: [...], style_opts: {...}}"""
    job_id = payload.get("job_id")
    job = JOBS.get(job_id)
    if not job or "src" not in job:
        raise HTTPException(404, "Unknown job — transcribe first.")

    segments = payload.get("segments") or job.get("segments") or []
    style_opts = payload.get("style_opts") or {}
    src = Path(job["src"])
    out = src.with_name(src.stem + "_captioned" + src.suffix)

    job["status"] = "rendering"
    job["output"] = None
    job["error"] = None

    def log(msg):
        print(f"[{job_id[:8]}] {msg}")
        job["log"].append(msg)

    async def _do():
        try:
            await asyncio.to_thread(
                cp.render_with_segments,
                src, segments, out, style_opts, log,
            )
            job["output"] = str(out)
            job["status"] = "done"
            log("Render complete.")
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            log(f"ERROR: {e}")

    asyncio.create_task(_do())
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return JSONResponse({
        "status": job["status"],
        "log": job["log"][-50:],
        "error": job["error"],
        "ready": bool(job["output"]) and job["status"] == "done",
    })


@app.get("/download/{job_id}")
def download(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("output"):
        raise HTTPException(404, "Not ready")
    path = Path(job["output"])
    return FileResponse(path, filename=path.name, media_type="video/mp4")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Auto-Caption</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0b0d10; color: #e8eaed; margin: 0; min-height: 100vh;
         padding: 24px; }
  .wrap { max-width: 1100px; margin: 0 auto; display: grid;
          grid-template-columns: 1fr 340px; gap: 22px; }
  @media (max-width: 780px) { .wrap { grid-template-columns: 1fr; } }
  .panel { background: #14181d; border: 1px solid #232a31; border-radius: 16px;
           padding: 22px; box-shadow: 0 10px 40px rgba(0,0,0,.4); }
  h1 { margin: 0 0 4px; font-size: 20px; letter-spacing: -0.01em; }
  h2 { margin: 0 0 12px; font-size: 14px; color: #a8b1bb; text-transform: uppercase;
       letter-spacing: 0.06em; font-weight: 600; }
  .sub { color: #8b96a1; font-size: 13px; margin-bottom: 18px; }

  /* Preview */
  .preview-wrap { position: relative; width: 100%; background: #000;
                  border-radius: 10px; overflow: hidden; aspect-ratio: 16/9;
                  border: 1px solid #232a31; }
  .preview-wrap video, .preview-wrap canvas, .preview-wrap .placeholder {
                  position: absolute; inset: 0; width: 100%; height: 100%;
                  object-fit: contain; }
  /* Timeline */
  .timeline { margin-top: 12px; padding: 10px 12px; background: #0a0d11;
              border: 1px solid #20262d; border-radius: 8px;
              display: none; }
  .timeline.visible { display: block; }
  .tl-row { display: flex; align-items: center; gap: 10px; font-size: 12px;
            color: #8b96a1; }
  .tl-time { font-family: ui-monospace, monospace; min-width: 86px;
             text-align: right; color: #cdd6df; }
  .tl-playbtn { width: 30px; height: 30px; border-radius: 50%;
                background: #5b8def; color: #fff; border: 0; cursor: pointer;
                display: flex; align-items: center; justify-content: center;
                font-size: 14px; padding: 0; margin: 0; }
  .tl-playbtn:hover { background: #4a7cdf; }
  .tl-track { position: relative; flex: 1; height: 36px; cursor: pointer;
              background: #14181d; border-radius: 6px; overflow: hidden; }
  .tl-seg { position: absolute; top: 5px; bottom: 5px;
            background: #2a3949; border: 1px solid #3a516a; border-radius: 3px;
            cursor: pointer; transition: .1s; min-width: 4px;
            overflow: hidden; padding: 2px 4px; font-size: 9px;
            white-space: nowrap; color: #cdd6df; }
  .tl-seg:hover { background: #3a516a; border-color: #5b8def; }
  .tl-seg.active { background: #5b8def; border-color: #79a5ff; color: #fff; }
  .tl-seg.has-pos { background: #4a8f56; border-color: #5fb96f; }
  .tl-seg.has-pos.active { background: #6ee08a; }
  .tl-playhead { position: absolute; top: 0; bottom: 0; width: 2px;
                 background: #ff5050; pointer-events: none; z-index: 2; }
  .tl-actions { display: flex; align-items: center; gap: 8px;
                margin-top: 8px; flex-wrap: wrap; }
  .tl-action-btn { background: #5b8def; color: #fff; border: 0;
                   border-radius: 5px; padding: 5px 10px; font-size: 11px;
                   font-weight: 600; cursor: pointer; transition: .12s; }
  .tl-action-btn:hover:not(:disabled) { background: #4a7cdf; }
  .tl-action-btn:disabled { background: #2a3038; color: #5b6470;
                            cursor: not-allowed; }
  .tl-action-btn.ghost { background: transparent; color: #8b96a1;
                         border: 1px solid #2a3038; }
  .tl-action-btn.ghost:hover:not(:disabled) { color: #cdd6df;
                                              border-color: #5b8def; }
  .tl-hint { font-size: 11px; color: #6c7681; margin-left: auto; }
  .seg-editor { margin-top: 10px; padding: 10px; background: #14181d;
                border: 1px solid #2a3038; border-radius: 6px; }
  .seg-editor-head { display: flex; align-items: center;
                     justify-content: space-between; margin-bottom: 6px;
                     font-size: 11px; color: #8b96a1; }
  #segEditorTitle { font-weight: 600; color: #cdd6df; }
  .seg-editor-x { background: transparent; color: #8b96a1; border: 0;
                  font-size: 16px; line-height: 1; cursor: pointer;
                  padding: 2px 6px; border-radius: 4px; }
  .seg-editor-x:hover { color: #cdd6df; background: #20262d; }
  #segEditorText { width: 100%; box-sizing: border-box; padding: 8px;
                   background: #0a0d11; color: #e6edf3; border: 1px solid
                   #2a3038; border-radius: 5px; font-family: inherit;
                   font-size: 13px; line-height: 1.4; resize: vertical; }
  #segEditorText:focus { outline: none; border-color: #5b8def; }
  .seg-editor-foot { display: flex; align-items: center; gap: 8px;
                     margin-top: 8px; }
  .seg-editor-hint { font-size: 10px; color: #6c7681; margin-right: auto; }
  .placeholder { display: flex; align-items: center; justify-content: center;
                 color: #4a5560; font-size: 14px; background:
                 linear-gradient(135deg, #0e1318 0%, #1a232c 100%); }
  /* Positioning shell — spans the available area; text-align centers the
     inline-block overlay horizontally without collapsing its shrink-to-fit
     width (which `left + translateX(-50%)` would do). */
  .overlay-anchor { position: absolute; pointer-events: none; }
  .caption-overlay { display: inline-block; pointer-events: auto;
                     cursor: grab;
                     font-family: Arial, sans-serif; white-space: pre;
                     text-align: center; line-height: 1.15;
                     user-select: none; touch-action: none;
                     box-sizing: border-box; vertical-align: top; }
  .caption-overlay.dragging { cursor: grabbing; }
  .caption-overlay::after { content: ""; position: absolute; inset: -4px;
                     border: 1px dashed transparent; border-radius: 4px;
                     pointer-events: none; transition: .12s; }
  .caption-overlay:hover::after { border-color: rgba(91,141,239,.45); }
  .drag-hint { position: absolute; top: 6px; left: 8px;
               background: rgba(0,0,0,.55); color: #cdd6df; font-size: 11px;
               padding: 3px 8px; border-radius: 4px; pointer-events: none;
               opacity: 0; transition: .15s; }
  .preview-wrap:hover .drag-hint { opacity: 1; }

  /* Drop zone */
  .drop { border: 2px dashed #2c343c; border-radius: 12px; padding: 24px 16px;
          text-align: center; cursor: pointer; transition: .15s;
          background: #0f1318; margin-top: 14px; }
  .drop.over { border-color: #5b8def; background: #131a25; }
  .drop strong { color: #cdd6df; }
  .drop small { color: #6c7681; display: block; margin-top: 4px; font-size: 12px; }
  .filename { color: #cdd6df; font-size: 12px; margin-top: 8px; min-height: 16px; }

  /* Controls */
  .group { margin-bottom: 18px; }
  .row { display: flex; gap: 10px; }
  .row > * { flex: 1; }
  label { font-size: 11px; color: #8b96a1; display: block; margin-bottom: 5px;
          text-transform: uppercase; letter-spacing: 0.04em; }
  select, input[type=number], input[type=text] {
          width: 100%; background: #0f1318; color: #e8eaed;
          border: 1px solid #2c343c; border-radius: 8px;
          padding: 8px 9px; font-size: 13px; }
  input[type=color] { width: 100%; height: 32px; background: #0f1318;
          border: 1px solid #2c343c; border-radius: 8px; padding: 2px;
          cursor: pointer; }
  input[type=range] { width: 100%; }
  .check { display: flex; align-items: center; gap: 8px; font-size: 13px;
           color: #cdd6df; margin: 6px 0; cursor: pointer; }
  .check input { margin: 0; }

  /* Button group (vertical + horizontal alignment) */
  .btn-row { display: flex; gap: 4px; background: #0a0d11;
             border: 1px solid #2c343c; border-radius: 8px; padding: 4px;
             margin-bottom: 8px; }
  .btn-row button { flex: 1; background: transparent; color: #cdd6df;
             border: 1px solid transparent; padding: 7px 8px; border-radius: 6px;
             font-size: 12px; font-weight: 500; cursor: pointer;
             margin-top: 0; transition: .12s; display: flex;
             align-items: center; justify-content: center; gap: 5px; }
  .btn-row button:hover { background: #1a2129; }
  .btn-row button.active { background: #5b8def; color: #fff;
             border-color: #79a5ff; }
  .btn-row .icon { width: 14px; height: 14px; opacity: .9; }

  /* Sliders with value display */
  .slider-row { display: flex; align-items: center; gap: 8px; }
  .slider-row input[type=range] { flex: 1; }
  .slider-row .val { font-family: ui-monospace, monospace; font-size: 12px;
                     color: #8b96a1; min-width: 38px; text-align: right; }

  button { width: 100%; margin-top: 8px; background: #5b8def; color: #fff;
           border: 0; padding: 12px 16px; border-radius: 10px; font-size: 15px;
           font-weight: 600; cursor: pointer; transition: .15s; }
  button:hover:not(:disabled) { background: #4a7cdf; }
  button:disabled { opacity: .55; cursor: not-allowed; }

  .log { margin-top: 14px; background: #0a0d11; border: 1px solid #20262d;
         border-radius: 8px; padding: 11px 13px; font-family: ui-monospace, monospace;
         font-size: 11px; max-height: 160px; overflow-y: auto; white-space: pre-wrap;
         color: #b6bec7; display: none; }
  .log.visible { display: block; }
  .ready { margin-top: 12px; padding: 12px; background: #0f2b1a;
           border: 1px solid #1f5234; border-radius: 8px; display: none;
           font-size: 13px; }
  .ready.visible { display: block; }
  .ready a { color: #6ee08a; font-weight: 600; text-decoration: none; }
  .err { color: #ff7a7a; margin-top: 8px; font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">

  <div class="panel">
    <h1>🎬 Auto-Caption</h1>
    <div class="sub">Drop a video → Transcribe → preview captions on the timeline → Render.</div>

    <div class="preview-wrap" id="previewWrap">
      <div class="placeholder" id="placeholder">No video loaded yet — drop one below</div>
      <video id="video" hidden playsinline preload="metadata"></video>
      <canvas id="canvas" hidden></canvas>
      <div class="overlay-anchor" id="overlayAnchor">
        <div class="caption-overlay" id="overlay">Sample caption text appears here</div>
      </div>
      <div class="drag-hint">Drag to position freely · grid to reset</div>
    </div>

    <div class="timeline" id="timeline">
      <div class="tl-row">
        <button class="tl-playbtn" id="playBtn" title="Play / Pause">▶</button>
        <div class="tl-track" id="tlTrack">
          <div class="tl-playhead" id="tlPlayhead" style="left:0"></div>
        </div>
        <span class="tl-time" id="tlTime">0:00 / 0:00</span>
      </div>

      <!-- Inline caption editor — opened by double-clicking a timeline block. -->
      <div class="seg-editor" id="segEditor" hidden>
        <div class="seg-editor-head">
          <span id="segEditorTitle">Edit caption</span>
          <button type="button" class="seg-editor-x" id="segEditorClose"
                  title="Close without saving">×</button>
        </div>
        <textarea id="segEditorText" rows="3"
                  placeholder="Caption text — use line breaks where you want lines to wrap."></textarea>
        <div class="seg-editor-foot">
          <span class="seg-editor-hint">⌘/Ctrl+Enter to save · Esc to cancel</span>
          <button type="button" class="tl-action-btn ghost" id="segEditorCancel">Cancel</button>
          <button type="button" class="tl-action-btn" id="segEditorSave">Save</button>
        </div>
      </div>
      <div class="tl-actions">
        <button id="applyPosAll" type="button" class="tl-action-btn"
                title="Copy the current caption's position to every segment">
          Apply position to all
        </button>
        <button id="clearPosAll" type="button" class="tl-action-btn ghost"
                title="Remove custom positions from every segment">
          Clear all positions
        </button>
        <span id="applyPosHint" class="tl-hint">
          Click a caption block to jump · drag overlay to set its position
        </span>
      </div>
    </div>

    <div class="drop" id="drop">
      <strong>Drop a video here</strong>
      <small>or click to choose — .mp4, .mov, .mkv, .webm</small>
      <input type="file" id="file" accept="video/*" hidden>
    </div>
    <div class="filename" id="filename"></div>

    <div class="log" id="log"></div>
    <div class="ready" id="ready"></div>
    <div class="err" id="err"></div>
  </div>

  <div class="panel">
    <h2>Style</h2>

    <div class="group">
      <label>Vertical position</label>
      <div class="btn-row" id="vRow">
        <button data-v="top">⬆ Top</button>
        <button data-v="mid">— Middle</button>
        <button data-v="bot" class="active">⬇ Bottom</button>
      </div>
      <label>Text alignment</label>
      <div class="btn-row" id="hRow">
        <button data-h="left">⟸ Left</button>
        <button data-h="center" class="active">↔ Center</button>
        <button data-h="right">⟹ Right</button>
      </div>
    </div>

    <div class="group">
      <label>Vertical margin <span class="val" id="mvVal">40 px</span></label>
      <input type="range" id="margin_v" min="0" max="400" value="40">

      <label style="margin-top:10px;">Horizontal margin <span class="val" id="mhVal">20 px</span></label>
      <input type="range" id="margin_h" min="0" max="400" value="20">
    </div>

    <div class="group">
      <label>Font family</label>
      <select id="font_name">
        <option>Arial</option>
        <option>Helvetica</option>
        <option>Helvetica Neue</option>
        <option>Times New Roman</option>
        <option>Georgia</option>
        <option>Courier New</option>
        <option>Menlo</option>
        <option>Verdana</option>
        <option>Trebuchet MS</option>
        <option>Comic Sans MS</option>
        <option>Impact</option>
      </select>

      <div class="row" style="margin-top:10px;">
        <div>
          <label>Size <span class="val" id="fsVal">24</span></label>
          <input type="range" id="font_size" min="10" max="80" value="24">
        </div>
        <div id="outlineSliderWrap">
          <label>Outline <span class="val" id="owVal">2</span></label>
          <input type="range" id="outline_width" min="0" max="8" value="2">
        </div>
      </div>

      <div class="row" style="margin-top:6px;">
        <label class="check"><input type="checkbox" id="bold"> Bold</label>
        <label class="check"><input type="checkbox" id="italic"> Italic</label>
      </div>
    </div>

    <div class="group">
      <div class="row">
        <div>
          <label>Text color</label>
          <input type="color" id="primary_color" value="#FFFFFF">
        </div>
        <div id="outlineColorWrap">
          <label>Outline color</label>
          <input type="color" id="outline_color" value="#000000">
        </div>
      </div>

      <label class="check" style="margin-top:10px;">
        <input type="checkbox" id="show_box"> Background box
      </label>
      <div id="boxControls" style="display:none;">
        <div class="row">
          <div>
            <label>Box color</label>
            <input type="color" id="back_color" value="#000000">
          </div>
          <div>
            <label>Opacity <span class="val" id="baVal">50%</span></label>
            <input type="range" id="back_alpha" min="0" max="255" value="128">
          </div>
        </div>
        <label style="margin-top:8px;">Padding <span class="val" id="bpVal">8 px</span></label>
        <input type="range" id="box_padding" min="0" max="40" value="8">
      </div>
    </div>

    <h2 style="margin-top:18px;">Text layout</h2>
    <div class="group">
      <div>
        <label>Max caption width <span class="val" id="wpctVal">100%</span></label>
        <input type="range" id="wrap_pct" min="20" max="100" value="100">
        <div style="font-size:11px;color:#7e8a99;margin-top:2px;">
          Caps the text/box width. 100% = use Horizontal margin only.
        </div>
      </div>
      <div class="row" style="margin-top:8px;">
        <div>
          <label>Max chars / line <span class="val" id="mcVal">42</span></label>
          <input type="range" id="max_chars" min="12" max="80" value="42">
        </div>
        <div>
          <label>Max lines <span class="val" id="mlVal">2</span></label>
          <input type="range" id="max_lines" min="1" max="4" value="2">
        </div>
      </div>
    </div>

    <h2 style="margin-top:18px;">Transcription</h2>
    <div class="group">
      <label>Whisper model</label>
      <select id="model">
        <option value="tiny">tiny — fastest</option>
        <option value="base" selected>base — balanced</option>
        <option value="small">small — better</option>
        <option value="medium">medium — slow</option>
        <option value="large">large — best</option>
      </select>
      <label class="check" style="margin-top:10px;">
        <input type="checkbox" id="refine" checked> Refine with Claude
      </label>
    </div>

    <button id="go" disabled>Transcribe</button>
    <button id="renderBtn" disabled style="margin-top:6px; background:#2a8c44; display:none;">Render captioned video</button>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const drop = $('drop'), fileInput = $('file'), filenameEl = $('filename');
const goBtn = $('go'), logEl = $('log'), readyEl = $('ready'), errEl = $('err');
const video = $('video'), canvas = $('canvas'), placeholder = $('placeholder');
const overlay = $('overlay'), previewWrap = $('previewWrap');
const overlayAnchor = $('overlayAnchor');
let chosen = null;

// ---------- file selection + preview frame ----------
function setFile(f) {
  chosen = f;
  filenameEl.textContent = f ? `${f.name} — ${(f.size/1e6).toFixed(1)} MB` : '';
  goBtn.disabled = !f;
  if (!f) return;
  // Show a quick preview frame from the local file before we upload, so the
  // user can style ahead of time. Once transcribed, we'll switch to the
  // server-streamed copy (which supports seeking).
  const url = URL.createObjectURL(f);
  video.src = url;
  video.hidden = false;
  placeholder.style.display = 'none';
  video.onloadedmetadata = () => {
    if (!serverVideoLoaded) {
      video.currentTime = Math.min(1, video.duration * 0.25);
    }
    updateOverlay();
  };
}

drop.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => setFile(e.target.files[0]));
['dragenter','dragover'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.add('over');
}));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.remove('over');
}));
drop.addEventListener('drop', e => {
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});

// ---------- style state ----------
// We track vertical and horizontal alignment independently, then derive the
// libass numpad value (1-9) from them. This makes the UI clearer (separate
// "vertical position" and "text alignment" controls) without changing the
// underlying single Alignment param libass expects.
let vAlign = 'bot';  // 'top' | 'mid' | 'bot'
let hAlign = 'center';  // 'left' | 'center' | 'right'
let customPos = null;  // {x, y} in video pixel space, or null when using alignment

// ---------- timeline / transcription state ----------
let serverVideoLoaded = false;  // true once we've swapped to the streamed video
let currentJobId = null;
let segments = [];  // [{start, end, text, pos_x?, pos_y?}]
let activeSegIdx = -1;
const SAMPLE_TEXT = 'Sample caption text appears here for live preview of style';

function currentAlignment() {
  // libass numpad: 1=BL 2=BC 3=BR 4=ML 5=MC 6=MR 7=TL 8=TC 9=TR
  const rowOffset = vAlign === 'top' ? 6 : vAlign === 'mid' ? 3 : 0;
  const col = hAlign === 'left' ? 1 : hAlign === 'right' ? 3 : 2;
  return rowOffset + col;
}

function alignToCss(a) {
  const row = Math.ceil(a / 3); // 1 = bottom, 2 = mid, 3 = top
  const col = ((a - 1) % 3); // 0 = left, 1 = center, 2 = right
  return {
    isBottom: row === 1, isMiddle: row === 2, isTop: row === 3,
    isLeft: col === 0, isCenter: col === 1, isRight: col === 2,
    textAlign: col === 0 ? 'left' : col === 2 ? 'right' : 'center',
  };
}

function setVAlign(v) {
  vAlign = v;
  $('vRow').querySelectorAll('button').forEach(b =>
    b.classList.toggle('active', b.dataset.v === v));
  customPos = null;
  updateOverlay();
}
function setHAlign(h) {
  hAlign = h;
  $('hRow').querySelectorAll('button').forEach(b =>
    b.classList.toggle('active', b.dataset.h === h));
  customPos = null;
  updateOverlay();
}

function wrapTextForPreview(text, maxChars) {
  const words = text.split(/\s+/);
  if (!words.length) return text;
  const lines = []; let cur = words[0];
  for (let i = 1; i < words.length; i++) {
    if (cur.length + 1 + words[i].length <= maxChars) cur += ' ' + words[i];
    else { lines.push(cur); cur = words[i]; }
  }
  lines.push(cur);
  return lines.join('\n');
}

// Match the server's wrap logic exactly: server uses `_rewrap_to_pct` if
// wrap_pct < 100, otherwise leaves the transcribe-time `apply_text_layout`
// (max_chars) wrap in place. To make the preview's box width identical to
// the burn, we re-wrap segments here using the same character budget.
function applyWrapMatchingServer(rawText, {fontSize, wrapPct, maxChars, maxLines, videoW, videoH}) {
  // Flatten any existing breaks — we recompute breaks from scratch.
  const flat = rawText.replace(/\s+/g, ' ').trim();
  if (!flat) return '';
  let charBudget = maxChars;
  if (wrapPct && wrapPct < 100) {
    const avgCharPx = Math.max(1, fontSize * (videoH / 288) * 0.55);
    const widthPx = Math.max(1, (wrapPct / 100) * videoW);
    charBudget = Math.max(6, Math.min(maxChars, Math.floor(widthPx / avgCharPx)));
  }
  const words = flat.split(' ');
  const lines = []; let cur = words[0];
  for (let i = 1; i < words.length; i++) {
    if (cur.length + 1 + words[i].length <= charBudget) cur += ' ' + words[i];
    else { lines.push(cur); cur = words[i]; }
  }
  lines.push(cur);
  // Note: maxLines is only used for the pre-transcription sample preview;
  // for real segments the server doesn't drop lines, so we don't either.
  if (!segments.length && maxLines && lines.length > maxLines) {
    return lines.slice(0, maxLines).join('\n');
  }
  return lines.join('\n');
}

function updateOverlay() {
  const fontSize = +$('font_size').value;
  const fontName = $('font_name').value;
  const outlineW = +$('outline_width').value;
  const primary = $('primary_color').value;
  const outline = $('outline_color').value;
  const mv = +$('margin_v').value;
  const mh = +$('margin_h').value;
  const bold = $('bold').checked;
  const italic = $('italic').checked;
  const showBox = $('show_box').checked;
  const backColor = $('back_color').value;
  const backAlpha = +$('back_alpha').value;
  const boxPadding = +$('box_padding').value;
  const maxChars = +$('max_chars').value;
  const maxLines = +$('max_lines').value;
  const wrapPct = +$('wrap_pct').value;

  $('fsVal').textContent = fontSize;
  $('owVal').textContent = outlineW;
  $('mvVal').textContent = mv + ' px';
  $('mhVal').textContent = mh + ' px';
  $('baVal').textContent = Math.round(backAlpha / 255 * 100) + '%';
  $('bpVal').textContent = boxPadding + ' px';
  $('mcVal').textContent = maxChars;
  $('mlVal').textContent = maxLines;
  $('wpctVal').textContent = wrapPct + '%';
  $('boxControls').style.display = showBox ? 'block' : 'none';
  // Outline + box now both supported (two-pass render on the server when both
  // are wanted), so keep the outline controls visible in box mode too.

  // If we have real segments after transcription, show the active one.
  // Otherwise show a sample for styling purposes.
  let rawText;
  if (segments.length) {
    rawText = activeSegIdx >= 0 ? segments[activeSegIdx].text : '';
  } else {
    rawText = SAMPLE_TEXT;
  }
  // Apply the same wrap the server will apply at render time. This is the
  // key to making the box width match the burn: both sides choose the SAME
  // line break points, so the longest line — and therefore the box width —
  // is identical in both.
  const displayText = applyWrapMatchingServer(rawText, {
    fontSize, wrapPct, maxChars, maxLines,
    videoW: video.videoWidth || canvas.width || 640,
    videoH: video.videoHeight || canvas.height || 360,
  });
  overlay.textContent = displayText;
  overlay.style.visibility = displayText ? 'visible' : 'hidden';

  // Per-segment position override takes precedence over global custom pos.
  let posForOverlay = customPos;
  if (activeSegIdx >= 0 && segments[activeSegIdx].pos_x != null) {
    posForOverlay = {x: segments[activeSegIdx].pos_x, y: segments[activeSegIdx].pos_y};
  }

  const a = alignToCss(currentAlignment());

  // libass renders captions in "script units" where PlayResY = 288 by default
  // (and our ASS writer also uses 288). FontSize / Outline / MarginV / MarginH
  // / Box-padding are all in those units. To preview them in CSS pixels we
  // need: css_px = script_units * (displayed_image_height / 288).
  const previewRect = previewWrap.getBoundingClientRect();
  const videoH = video.videoHeight || canvas.height || 360;
  const videoW = video.videoWidth || canvas.width || 640;
  const previewH = previewRect.height;
  const previewW = previewRect.width;
  // The image is `object-fit: contain`, so the actual displayed image rect
  // may be letterboxed inside the preview area.
  const vidAspect = videoW / videoH;
  const dispAspect = previewW / previewH;
  let imgW, imgH;
  if (vidAspect > dispAspect) { imgW = previewW; imgH = previewW / vidAspect; }
  else { imgH = previewH; imgW = previewH * vidAspect; }
  const offsetX = (previewW - imgW) / 2;
  const offsetY = (previewH - imgH) / 2;
  const scale = imgH / videoH;            // video px → preview px
  const scriptToCss = imgH / 288;          // script units → preview px
  const fontPx = fontSize * scriptToCss;
  const outlinePx = outlineW * scriptToCss;
  const marginVpx = mv * scriptToCss;
  const marginHpx = mh * scriptToCss;

  overlay.style.fontFamily = `"${fontName}", sans-serif`;
  overlay.style.fontSize = fontPx + 'px';
  overlay.style.color = primary;
  overlay.style.fontWeight = bold ? '700' : '400';
  overlay.style.fontStyle = italic ? 'italic' : 'normal';
  overlay.style.textAlign = a.textAlign;

  // Stroke: use -webkit-text-stroke with paint-order so the stroke draws
  // behind the fill (closer match to libass than chunky multi-shadow stacks).
  // The server runs a two-pass render when both box + outline are on, so the
  // preview also shows both layered.
  if (outlinePx > 0) {
    overlay.style.webkitTextStroke = (outlinePx * 2) + 'px ' + outline;
    overlay.style.paintOrder = 'stroke fill';
    overlay.style.textShadow = 'none';
  } else {
    overlay.style.webkitTextStroke = '';
    overlay.style.paintOrder = '';
    overlay.style.textShadow = 'none';
  }

  if (showBox) {
    const alphaCss = backAlpha / 255;
    overlay.style.backgroundColor = hexToRgba(backColor, alphaCss);
    // box padding in libass is the `Outline` value (script units) — match preview
    overlay.style.padding = (boxPadding * scriptToCss) + 'px';
  } else {
    overlay.style.backgroundColor = 'transparent';
    overlay.style.padding = '0';
  }

  // Reset both elements
  overlay.style.maxWidth = '';
  overlay.style.transform = '';
  for (const p of ['left','right','top','bottom','transform','width','height','textAlign']) {
    overlayAnchor.style[p] = '';
  }

  // Cap the overlay width to the visible video area. We do NOT subtract
  // marginH from this — libass with WrapStyle=2 ignores margins for wrap
  // and so do we. Wrap width is controlled by max_chars + wrap_pct, both
  // applied above when computing `displayText`. This cap only keeps the
  // preview overlay from drawing outside the preview frame.
  overlay.style.maxWidth = Math.max(40, imgW) + 'px';

  // Position the ANCHOR (full-width or full-height strip) and let the inner
  // inline-block overlay shrink-to-fit horizontally — that lets the box hug
  // the text the way libass does, instead of collapsing the available
  // wrapping width like `left + translateX(-50%)` would.
  if (posForOverlay) {
    // Custom drag position: anchor is a zero-width pin at the drag point,
    // overlay centers around it via its own translate(-50%, -50%).
    const cssX = offsetX + posForOverlay.x * scale;
    const cssY = offsetY + posForOverlay.y * scale;
    overlayAnchor.style.left = cssX + 'px';
    overlayAnchor.style.top = cssY + 'px';
    overlayAnchor.style.width = '0';
    overlayAnchor.style.height = '0';
    overlayAnchor.style.textAlign = 'center';
    overlay.style.transform = 'translate(-50%, -50%)';
  } else {
    // Horizontal: anchor strip spans the video horizontally (minus margins
    // on the left/right), text-aligned to position the inline-block overlay.
    overlayAnchor.style.left  = (offsetX + marginHpx) + 'px';
    overlayAnchor.style.right = (previewW - (offsetX + imgW - marginHpx)) + 'px';
    overlayAnchor.style.textAlign = a.isLeft ? 'left' : (a.isRight ? 'right' : 'center');
    // Vertical:
    if (a.isBottom) overlayAnchor.style.bottom = (offsetY + marginVpx) + 'px';
    else if (a.isTop) overlayAnchor.style.top = (offsetY + marginVpx) + 'px';
    else { // middle
      overlayAnchor.style.top = (offsetY + imgH / 2) + 'px';
      overlay.style.transform = 'translateY(-50%)';
    }
  }
}

// ---------- drag-to-position ----------
function getImageRect() {
  const r = previewWrap.getBoundingClientRect();
  const videoH = video.videoHeight || canvas.height || 360;
  const videoW = video.videoWidth || canvas.width || 640;
  const vidA = videoW / videoH, dispA = r.width / r.height;
  let imgW, imgH;
  if (vidA > dispA) { imgW = r.width; imgH = r.width / vidA; }
  else { imgH = r.height; imgW = r.height * vidA; }
  return { r, imgW, imgH, offsetX: (r.width - imgW)/2, offsetY: (r.height - imgH)/2,
           scale: imgH / videoH, videoH, videoW };
}

let dragging = false, dragOffset = null, dragResumePlay = false;
overlay.addEventListener('pointerdown', (e) => {
  if (!chosen) return; // need a video first
  dragging = true;
  // Pause playback during drag so the caption text doesn't swap segments
  // (and re-wrap) underneath the user — would look like the box is
  // "changing size while I move it."
  if (video && !video.paused) { dragResumePlay = true; video.pause(); }
  else { dragResumePlay = false; }
  overlay.classList.add('dragging');
  overlay.setPointerCapture(e.pointerId);
  const ob = overlay.getBoundingClientRect();
  dragOffset = { dx: e.clientX - (ob.left + ob.width/2), dy: e.clientY - (ob.top + ob.height/2) };
  e.preventDefault();
});
overlay.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const info = getImageRect();
  const localX = e.clientX - info.r.left - dragOffset.dx;
  const localY = e.clientY - info.r.top - dragOffset.dy;
  const vidX = Math.max(0, Math.min(info.videoW,
                          (localX - info.offsetX) / info.scale));
  const vidY = Math.max(0, Math.min(info.videoH,
                          (localY - info.offsetY) / info.scale));
  if (activeSegIdx >= 0) {
    // We have real segments: this drag sets THIS segment's position only.
    segments[activeSegIdx].pos_x = Math.round(vidX);
    segments[activeSegIdx].pos_y = Math.round(vidY);
    renderTimeline();
  } else {
    customPos = { x: vidX, y: vidY };
  }
  updateOverlay();
});
overlay.addEventListener('pointerup', (e) => {
  if (!dragging) return;
  dragging = false;
  overlay.classList.remove('dragging');
  try { overlay.releasePointerCapture(e.pointerId); } catch {}
  if (dragResumePlay) { video.play().catch(()=>{}); dragResumePlay = false; }
});

function hexToRgba(hex, a) {
  const h = hex.replace('#','');
  const r = parseInt(h.slice(0,2),16), g = parseInt(h.slice(2,4),16), b = parseInt(h.slice(4,6),16);
  return `rgba(${r},${g},${b},${a})`;
}

// Bind controls
['font_size','font_name','outline_width','primary_color','outline_color',
 'margin_v','margin_h','bold','italic','show_box','back_color','back_alpha',
 'box_padding','max_chars','max_lines','wrap_pct']
  .forEach(id => { $(id).addEventListener('input', updateOverlay); $(id).addEventListener('change', updateOverlay); });

$('vRow').querySelectorAll('button').forEach(b =>
  b.addEventListener('click', () => setVAlign(b.dataset.v)));
$('hRow').querySelectorAll('button').forEach(b =>
  b.addEventListener('click', () => setHAlign(b.dataset.h)));

// ---------- timeline ----------
function renderTimeline() {
  const track = $('tlTrack');
  // Remove old segment blocks (keep playhead)
  track.querySelectorAll('.tl-seg').forEach(n => n.remove());
  if (!segments.length || !video.duration || !isFinite(video.duration)) return;
  const dur = video.duration;
  segments.forEach((s, i) => {
    const left = (s.start / dur) * 100;
    const width = Math.max(0.4, ((s.end - s.start) / dur) * 100);
    const el = document.createElement('div');
    el.className = 'tl-seg' + (i === activeSegIdx ? ' active' : '')
                            + (s.pos_x != null ? ' has-pos' : '');
    el.style.left = left + '%';
    el.style.width = width + '%';
    el.textContent = s.text.replace(/\n/g, ' ').slice(0, 40);
    el.title = `[${formatTime(s.start)} → ${formatTime(s.end)}] ${s.text}`
               + (s.pos_x != null ? `\nposition: (${s.pos_x}, ${s.pos_y}) — right-click to clear` : '');
    el.addEventListener('click', () => {
      video.currentTime = s.start + 0.01;
      if (video.paused) video.play().catch(() => {});
    });
    el.addEventListener('dblclick', (e) => {
      e.preventDefault();
      e.stopPropagation();
      openSegmentEditor(i);
    });
    el.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      if (s.pos_x != null) {
        delete s.pos_x; delete s.pos_y;
        renderTimeline(); updateOverlay();
      }
    });
    track.appendChild(el);
  });
}

function formatTime(t) {
  if (!isFinite(t)) return '0:00';
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2,'0')}`;
}

// ---------- inline caption editor ----------
let editingSegIdx = -1;
function openSegmentEditor(i) {
  if (i < 0 || i >= segments.length) return;
  editingSegIdx = i;
  const s = segments[i];
  // Pause playback so the user can read while editing — and so the timeline
  // doesn't advance the active segment under them.
  if (video && !video.paused) video.pause();
  // Jump to this segment so the preview shows what they're editing.
  video.currentTime = s.start + 0.01;
  activeSegIdx = i;
  $('segEditorTitle').textContent =
    `Edit caption ${i + 1} of ${segments.length}  ·  ${formatTime(s.start)} → ${formatTime(s.end)}`;
  $('segEditorText').value = s.text;
  $('segEditor').hidden = false;
  // Defer focus so the textarea can size correctly first.
  setTimeout(() => {
    const ta = $('segEditorText');
    ta.focus();
    ta.select();
  }, 0);
  updateOverlay();
}
function closeSegmentEditor() {
  editingSegIdx = -1;
  $('segEditor').hidden = true;
}
function saveSegmentEditor() {
  if (editingSegIdx < 0) return;
  const newText = $('segEditorText').value.trim();
  if (!newText) {
    if (!confirm('Caption text is empty. Remove this caption?')) return;
    segments.splice(editingSegIdx, 1);
  } else {
    segments[editingSegIdx] = {...segments[editingSegIdx], text: newText};
  }
  closeSegmentEditor();
  renderTimeline();
  updateOverlay();
  logEl.textContent += 'Caption updated. Click "Render captioned video" to re-burn.\n';
}
$('segEditorClose').addEventListener('click', closeSegmentEditor);
$('segEditorCancel').addEventListener('click', closeSegmentEditor);
$('segEditorSave').addEventListener('click', saveSegmentEditor);
$('segEditorText').addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { e.preventDefault(); closeSegmentEditor(); }
  else if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    saveSegmentEditor();
  }
});

function updateActiveSegment() {
  if (!segments.length) { activeSegIdx = -1; return; }
  const t = video.currentTime;
  const idx = segments.findIndex(s => t >= s.start && t < s.end);
  if (idx !== activeSegIdx) {
    activeSegIdx = idx;
    renderTimeline();
  }
}

video.addEventListener('timeupdate', () => {
  const dur = video.duration || 0;
  const t = video.currentTime;
  if (dur > 0) {
    $('tlPlayhead').style.left = ((t / dur) * 100) + '%';
  }
  $('tlTime').textContent = `${formatTime(t)} / ${formatTime(dur)}`;
  updateActiveSegment();
  updateOverlay();
});
video.addEventListener('loadedmetadata', () => {
  $('tlTime').textContent = `0:00 / ${formatTime(video.duration)}`;
  renderTimeline();
});
video.addEventListener('play', () => $('playBtn').textContent = '❚❚');
video.addEventListener('pause', () => $('playBtn').textContent = '▶');

$('playBtn').addEventListener('click', () => {
  if (video.paused) video.play().catch(() => {});
  else video.pause();
});

$('tlTrack').addEventListener('click', (e) => {
  // Clicks that hit a .tl-seg are handled there; this catches empty track.
  if (e.target.classList.contains('tl-seg')) return;
  const r = $('tlTrack').getBoundingClientRect();
  const ratio = (e.clientX - r.left) / r.width;
  if (video.duration) video.currentTime = ratio * video.duration;
});

// Resolve the "current position" — the one we'd apply to every segment.
// Priority: active segment's pos > any segment with a pos > customPos.
function resolveCurrentPos() {
  if (activeSegIdx >= 0 && segments[activeSegIdx]?.pos_x != null) {
    return { x: segments[activeSegIdx].pos_x, y: segments[activeSegIdx].pos_y };
  }
  const withPos = segments.find(s => s.pos_x != null);
  if (withPos) return { x: withPos.pos_x, y: withPos.pos_y };
  if (customPos) return { x: Math.round(customPos.x), y: Math.round(customPos.y) };
  return null;
}

$('applyPosAll').addEventListener('click', () => {
  if (!segments.length) return;
  const pos = resolveCurrentPos();
  if (!pos) {
    alert('Drag the caption to a position first, then click "Apply position to all".');
    return;
  }
  segments.forEach(s => { s.pos_x = pos.x; s.pos_y = pos.y; });
  // Once segments carry positions, the global drag-position is redundant.
  customPos = null;
  renderTimeline();
  updateOverlay();
  logEl.textContent += `Applied position (${pos.x}, ${pos.y}) to all ${segments.length} segments.\n`;
});

$('clearPosAll').addEventListener('click', () => {
  if (!segments.length) return;
  let cleared = 0;
  segments.forEach(s => {
    if (s.pos_x != null) { delete s.pos_x; delete s.pos_y; cleared++; }
  });
  customPos = null;
  renderTimeline();
  updateOverlay();
  if (cleared) logEl.textContent += `Cleared positions on ${cleared} segments.\n`;
});

window.addEventListener('resize', updateOverlay);
updateOverlay();

// ---------- submit + poll ----------
goBtn.addEventListener('click', async () => {
  if (!chosen) return;
  goBtn.disabled = true;
  errEl.textContent = '';
  readyEl.classList.remove('visible');
  logEl.classList.add('visible');
  logEl.textContent = 'Uploading & transcribing... (this can take a minute)\n';

  const fd = new FormData();
  fd.append('file', chosen);
  fd.append('model', $('model').value);
  fd.append('refine', $('refine').checked);
  fd.append('max_chars', $('max_chars').value);
  fd.append('max_lines', $('max_lines').value);

  let data;
  try {
    const resp = await fetch('/transcribe', { method: 'POST', body: fd });
    if (!resp.ok) throw new Error(await resp.text());
    data = await resp.json();
  } catch (e) {
    errEl.textContent = 'Transcribe failed: ' + e.message;
    goBtn.disabled = false;
    return;
  }

  currentJobId = data.job_id;
  segments = data.segments || [];
  logEl.textContent += `Transcribed ${segments.length} segments. Now playable on the timeline.\n`;

  // Swap to the server-streamed video (supports proper seeking on any format)
  serverVideoLoaded = true;
  video.src = data.video_url;
  video.controls = false;  // we use our custom controls now
  video.load();

  $('timeline').classList.add('visible');
  $('renderBtn').style.display = 'block';
  $('renderBtn').disabled = false;
  goBtn.textContent = 'Re-transcribe';
  goBtn.disabled = false;
  // Force overlay refresh now that we have segments
  setTimeout(() => { renderTimeline(); updateOverlay(); }, 100);
});

$('renderBtn').addEventListener('click', async () => {
  if (!currentJobId) return;
  $('renderBtn').disabled = true;
  errEl.textContent = '';
  readyEl.classList.remove('visible');
  logEl.classList.add('visible');
  logEl.textContent = 'Rendering captioned video...\n';

  const style_opts = {
    font_size: +$('font_size').value,
    font_name: $('font_name').value,
    primary_color: $('primary_color').value,
    outline_color: $('outline_color').value,
    outline_width: +$('outline_width').value,
    alignment: currentAlignment(),
    margin_v: +$('margin_v').value,
    margin_l: +$('margin_h').value,
    margin_r: +$('margin_h').value,
    border_style: $('show_box').checked ? 4 : 1,
    back_color: $('back_color').value,
    back_alpha: +$('back_alpha').value,
    bold: $('bold').checked,
    italic: $('italic').checked,
    box_padding: +$('box_padding').value,
    wrap_pct: +$('wrap_pct').value,
  };
  // Global default position from drag (when no segment was active during drag)
  if (customPos && !segments.some(s => s.pos_x != null)) {
    style_opts.pos_x = Math.round(customPos.x);
    style_opts.pos_y = Math.round(customPos.y);
  }

  try {
    const resp = await fetch('/render', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        job_id: currentJobId,
        segments: segments,
        style_opts: style_opts,
      }),
    });
    if (!resp.ok) throw new Error(await resp.text());
  } catch (e) {
    errEl.textContent = 'Render request failed: ' + e.message;
    $('renderBtn').disabled = false;
    return;
  }
  pollRender(currentJobId);
});

async function pollRender(jobId) {
  while (true) {
    await new Promise(r => setTimeout(r, 1500));
    let s;
    try {
      s = await fetch('/status/' + jobId).then(r => r.json());
    } catch (e) { continue; }
    logEl.textContent = s.log.join('\n');
    logEl.scrollTop = logEl.scrollHeight;
    if (s.status === 'done' && s.ready) {
      readyEl.innerHTML = `✅ Ready — <a href="/download/${jobId}" download>Download captioned video</a>`;
      readyEl.classList.add('visible');
      $('renderBtn').disabled = false;
      return;
    }
    if (s.status === 'error') {
      errEl.textContent = 'Error: ' + (s.error || 'unknown');
      $('renderBtn').disabled = false;
      return;
    }
  }
}


</script>
</body>
</html>"""


import time as _time
_APP_START_TIME = _time.time()


@app.get("/", response_class=HTMLResponse)
def index():
    # Stamp the rendered page with the server start time so a hard-reload
    # makes it obvious whether the browser has the latest, and force no-cache.
    stamped = INDEX_HTML.replace(
        "🎬 Auto-Caption</h1>",
        f"🎬 Auto-Caption</h1><div style='font-size:10px;color:#5b8def;margin-top:-3px;'>build {int(_APP_START_TIME)}</div>",
        1,
    )
    return HTMLResponse(
        stamped,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8765, reload=False)
