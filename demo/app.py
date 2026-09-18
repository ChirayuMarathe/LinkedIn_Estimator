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
    n_posts = MODEL.trained_on.get("n_posts", "?")
    n_companies = len(MODEL.company_stats)
    return (
        HTML.replace("{{PAGES}}", page_opts)
        .replace("{{FORMATS}}", fmt_opts)
        .replace("{{N_POSTS}}", str(n_posts))
        .replace("{{N_COMPANIES}}", str(n_companies))
    )


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LinkedIn Engagement Estimator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#f8f9fb; --card:#fff; --ink:#1a1d23; --muted:#6b7280;
    --line:#e5e7eb; --accent:#2563eb; --accent-h:#1d4ed8;
    --ok:#059669; --ok-bg:#ecfdf5;
    --sh:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
    --sh2:0 4px 14px rgba(0,0,0,.08);
  }
  @media(prefers-color-scheme:dark){:root{
    --bg:#0f1117; --card:#1a1d27; --ink:#e8eaed; --muted:#8b92a0;
    --line:#2a2d3a; --accent:#6390f0; --accent-h:#7ba4f7;
    --ok:#34d399; --ok-bg:rgba(52,211,153,.08);
    --sh:0 1px 3px rgba(0,0,0,.3); --sh2:0 4px 14px rgba(0,0,0,.4);
  }}
  *{box-sizing:border-box;margin:0}
  body{font:15px/1.6 'Inter',system-ui,sans-serif;background:var(--bg);color:var(--ink)}
  main{max-width:720px;margin:0 auto;padding:40px 20px}
  h1{font-size:24px;font-weight:700;letter-spacing:-.3px;margin-bottom:4px}
  p.sub{color:var(--muted);margin-bottom:28px;font-size:14px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px;margin-bottom:20px;box-shadow:var(--sh);transition:box-shadow .2s}
  .card:hover{box-shadow:var(--sh2)}
  label{display:block;font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:.4px;color:var(--muted);margin:16px 0 6px}
  label:first-child{margin-top:0}
  textarea,select,input{width:100%;padding:10px 14px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:inherit;font:inherit;font-size:14px;transition:border-color .2s,box-shadow .2s}
  select option{background:var(--card);color:var(--ink)}
  textarea:focus,select:focus,input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px rgba(37,99,235,.12)}
  textarea{min-height:130px;resize:vertical}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:560px){.row{grid-template-columns:1fr}}
  button{margin-top:20px;width:100%;background:var(--accent);color:#fff;border:0;border-radius:10px;padding:12px 20px;font-weight:600;font-size:15px;cursor:pointer;transition:background .2s,transform .1s}
  button:hover{background:var(--accent-h)} button:active{transform:scale(.98)}
  button:disabled{opacity:.7;cursor:wait}
  .hint{color:var(--muted);font-size:12px;letter-spacing:.2px}
  .res{border-color:var(--ok);animation:fadeUp .3s ease-out}
  @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .big{font-size:32px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.5px;margin:4px 0}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
  .ml{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;color:var(--muted)}
  #basis{margin-top:16px;padding-top:14px;border-top:1px solid var(--line);font-size:13px}
  #err{color:#ef4444;font-size:14px;font-weight:500}
  .ft{text-align:center;color:var(--muted);font-size:12px;margin-top:32px}
</style></head><body><main>
<h1>LinkedIn Engagement Estimator</h1>
<p class="sub">Paste a post, pick its format and page, get an estimated likes range and comment count.</p>
<form class="card" id="f">
  <label for="text">Post text</label>
  <textarea id="text" required placeholder="Write or paste your LinkedIn post here..."></textarea>
  <div class="row">
    <div><label for="media_type">Format</label><select id="media_type">{{FORMATS}}</select></div>
    <div><label for="num_images">Images</label><input id="num_images" type="number" min="0" max="20" value="0"></div>
  </div>
  <label for="company">Company page</label>
  <select id="company">{{PAGES}}<option value="">Other page (enter details below)</option></select>
  <div class="row" id="other" hidden>
    <div><label for="followers">Followers</label><input id="followers" type="number" min="0" placeholder="e.g. 50000"></div>
    <div><label for="baseline">Typical likes <span class="hint">(optional, improves accuracy)</span></label><input id="baseline" type="number" min="0" placeholder="e.g. 40"></div>
  </div>
  <button id="btn">Estimate Engagement</button>
</form>
<div class="card res" id="out" hidden>
  <div class="grid2">
    <div><div class="ml">\U0001f44d Likes (80% range)</div><div class="big" id="likes"></div><div class="hint" id="likes_pt"></div></div>
    <div><div class="ml">\U0001f4ac Comments</div><div class="big" id="comments"></div><div class="hint" id="comments_rng"></div></div>
  </div>
  <p class="hint" id="basis"></p>
</div>
<p id="err"></p>
<div class="ft">Trained on {{N_POSTS}} posts \u00b7 {{N_COMPANIES}} company pages</div>
<script>
const $=id=>document.getElementById(id);
$('company').onchange=()=>{$('other').hidden=$('company').value!=='';};
$('f').onsubmit=async e=>{
  e.preventDefault();$('err').textContent='';
  const btn=$('btn');btn.textContent='Estimating\u2026';btn.disabled=true;
  const body={text:$('text').value,media_type:$('media_type').value,num_images:+$('num_images').value||0};
  if($('company').value)body.company=$('company').value;
  else{body.company_followers=+$('followers').value||null;if($('baseline').value)body.baseline_likes=+$('baseline').value;}
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(!r.ok){$('err').textContent=d.detail?.toString()||'Error';$('out').hidden=true;return;}
    $('likes').textContent=`${d.likes.low.toLocaleString()} \u2013 ${d.likes.high.toLocaleString()}`;
    $('likes_pt').textContent=`point estimate: ${d.likes.estimate.toLocaleString()}`;
    $('comments').textContent=d.comments.estimate;
    $('comments_rng').textContent=`range: ${d.comments.low} \u2013 ${d.comments.high}`;
    const src={page_history:"this page's last 8 posts",user_baseline:'the typical likes you entered',cold_start_followers_only:'follower count only (add typical likes for better results)'}[d.baseline.source];
    $('basis').textContent=`Baseline: ${d.baseline.likes} likes / ${d.baseline.comments} comments, from ${src}.`;
    $('out').hidden=false;$('out').style.animation='none';$('out').offsetHeight;$('out').style.animation='';
  }finally{btn.textContent='Estimate Engagement';btn.disabled=false;}
};
</script></main></body></html>"""

