"""
FastAPI demo.

Run:   uvicorn demo.app:app --reload
Open:  http://127.0.0.1:8000        (web form)
       http://127.0.0.1:8000/docs   (interactive API docs)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from estimator.features import MEDIA_TYPES  # noqa: E402
from estimator.model import EngagementModel  # noqa: E402

app = FastAPI(title="LinkedIn Engagement Estimator", version="1.0")
MODEL = EngagementModel.load()


class PostIn(BaseModel):
    text: str = Field(..., min_length=1, description="Full post text")
    media_type: str = Field("text_only", description=f"One of: {', '.join(MEDIA_TYPES)}")
    num_images: int = Field(0, ge=0, le=20)
    company: str | None = Field(None, description="A page the model knows (see /pages)")
    company_followers: int | None = Field(None, ge=0, description="Needed for pages the model doesn't know")
    baseline_likes: float | None = Field(None, ge=0, description="Typical likes on this page's recent posts (strongly recommended for new pages)")
    baseline_comments: float | None = Field(None, ge=0)
    posted_at: datetime | None = Field(None, description="Planned publish time (UTC). Defaults to now.")


@app.get("/health")
def health():
    return {"status": "ok", "trained_on": MODEL.trained_on}


@app.get("/pages")
def pages():
    return MODEL.company_stats


@app.get("/formats")
def formats():
    return MEDIA_TYPES


@app.post("/predict")
def predict(post: PostIn):
    try:
        return MODEL.predict(**post.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def index():
    page_opts = "".join(
        f'<option value="{c}">{c} ({s["followers"]:,} followers)</option>' for c, s in sorted(MODEL.company_stats.items())
    )
    fmt_opts = "".join(f'<option value="{m}">{m.replace("_", " ")}</option>' for m in MEDIA_TYPES)
    return HTML.replace("{{PAGES}}", page_opts).replace("{{FORMATS}}", fmt_opts)


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LinkedIn Engagement Estimator</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1b1f24; --muted:#5b6470; --line:#dfe3e8; --accent:#0a66c2; }
  @media (prefers-color-scheme: dark) { :root { --bg:#111418; --card:#1a1f25; --ink:#e7eaee; --muted:#9aa4af; --line:#2c333b; --accent:#5aa7f0; } }
  * { box-sizing:border-box } body { margin:0; font:15px/1.5 system-ui,sans-serif; background:var(--bg); color:var(--ink); }
  main { max-width:760px; margin:0 auto; padding:24px 16px; }
  h1 { font-size:22px; margin:0 0 4px } p.sub { color:var(--muted); margin:0 0 20px }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; margin-bottom:16px }
  label { display:block; font-weight:600; margin:12px 0 4px } label:first-child { margin-top:0 }
  textarea, select, input { width:100%; padding:9px 10px; border:1px solid var(--line); border-radius:8px; background:transparent; color:inherit; font:inherit }
  textarea { min-height:140px; resize:vertical }
  .row { display:grid; grid-template-columns:1fr 1fr; gap:12px } @media (max-width:560px){ .row { grid-template-columns:1fr } }
  button { margin-top:16px; background:var(--accent); color:#fff; border:0; border-radius:8px; padding:10px 18px; font-weight:600; cursor:pointer }
  .hint { color:var(--muted); font-size:13px }
  .big { font-size:30px; font-weight:700; font-variant-numeric:tabular-nums }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px }
  #err { color:#c0392b }
</style></head><body><main>
<h1>LinkedIn Engagement Estimator</h1>
<p class="sub">Paste a post, pick its format and page, get an estimated likes range and comment count.</p>
<form class="card" id="f">
  <label for="text">Post text</label>
  <textarea id="text" required placeholder="Introducing ..."></textarea>
  <div class="row">
    <div><label for="media_type">Format</label><select id="media_type">{{FORMATS}}</select></div>
    <div><label for="num_images">Number of images</label><input id="num_images" type="number" min="0" max="20" value="0"></div>
  </div>
  <label for="company">Page</label>
  <select id="company">{{PAGES}}<option value="">Other page (enter details below)</option></select>
  <div class="row" id="other" hidden>
    <div><label for="followers">Followers</label><input id="followers" type="number" min="0"></div>
    <div><label for="baseline">Typical likes per post <span class="hint">(optional, much more accurate)</span></label><input id="baseline" type="number" min="0"></div>
  </div>
  <button>Estimate</button>
</form>
<div class="card" id="out" hidden>
  <div class="grid2">
    <div><div class="hint">Likes (80% range)</div><div class="big" id="likes"></div><div class="hint" id="likes_pt"></div></div>
    <div><div class="hint">Comments</div><div class="big" id="comments"></div><div class="hint" id="comments_rng"></div></div>
  </div>
  <p class="hint" id="basis"></p>
</div>
<p id="err"></p>
<script>
const $ = id => document.getElementById(id);
$('company').onchange = () => { $('other').hidden = $('company').value !== ''; };
$('f').onsubmit = async e => {
  e.preventDefault(); $('err').textContent = '';
  const body = { text: $('text').value, media_type: $('media_type').value, num_images: +$('num_images').value || 0 };
  if ($('company').value) body.company = $('company').value;
  else { body.company_followers = +$('followers').value || null; if ($('baseline').value) body.baseline_likes = +$('baseline').value; }
  const r = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const d = await r.json();
  if (!r.ok) { $('err').textContent = d.detail?.toString() || 'Error'; $('out').hidden = true; return; }
  $('likes').textContent = `${d.likes.low.toLocaleString()} - ${d.likes.high.toLocaleString()}`;
  $('likes_pt').textContent = `point estimate ${d.likes.estimate.toLocaleString()}`;
  $('comments').textContent = d.comments.estimate;
  $('comments_rng').textContent = `range ${d.comments.low} - ${d.comments.high}`;
  const src = { page_history: "this page's last 8 posts", user_baseline: 'the typical likes you entered', cold_start_followers_only: 'follower count only (low confidence: add typical likes)' }[d.baseline.source];
  $('basis').textContent = `Baseline: ${d.baseline.likes} likes / ${d.baseline.comments} comments, from ${src}.`;
  $('out').hidden = false;
};
</script></main></body></html>"""
