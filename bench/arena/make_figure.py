#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 AstraLibernis

"""Render bench/arena/scoreboard.json as a standalone HTML figure.

Form: a dumbbell for time (two entities per row, values spanning ~35x, so a
log axis), and a companion dot panel for the accuracy gap (four orders of
magnitude, also log). Two panels rather than one dual-axis plot: time in ms
and a dimensionless relative gap share no scale, and overlaying them would
invent a relationship that is not in the data.

Palette: dataviz reference slots 1 and 2, validated all-pairs in both modes
(light CVD dE 24.7 / normal 33.6; dark 26.8 / 31.8).
"""
import json, math
from pathlib import Path

ARENA = Path(__file__).resolve().parent
rows = json.loads((ARENA / "scoreboard.json").read_text())

W, LEFT, GAPW, RIGHT = 1060, 240, 250, 28
TRACK = W - LEFT - GAPW - RIGHT - 40
ROWH, TOP = 58, 104
H = TOP + ROWH * len(rows) + 26

TMIN, TMAX = 130, 8000
def tx(ms):
    return LEFT + (math.log10(ms) - math.log10(TMIN)) / (math.log10(TMAX) - math.log10(TMIN)) * TRACK

GMIN, GMAX = 1e-7, 1e-2
GX0 = LEFT + TRACK + 40
def gx(v):
    v = max(v, GMIN)
    return GX0 + (math.log10(v) - math.log10(GMIN)) / (math.log10(GMAX) - math.log10(GMIN)) * GAPW

def fmt_ms(v):
    return f"{v/1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"

def fmt_score(v, metric):
    return f"{v:,.0f}" if metric == "rmse" and v > 100 else f"{v:.6f}"

body, tbody = [], []
for i, r in enumerate(rows):
    y = TOP + i * ROWH + ROWH / 2
    z, s = r["zarbor"]["model_total"], r["ref"]["model_total"]
    zs, ss = r["zarbor_score"], r["ref_score"]
    ratio = s / z
    lo_first = z <= s
    better = (zs > ss) if r["metric"] == "auc" else (zs < ss)
    gap = abs(zs - ss) / abs(ss)

    body.append(f'<g class="row" data-i="{i}">')
    body.append(f'<rect class="hit" x="0" y="{y-ROWH/2:.0f}" width="{W}" height="{ROWH}"/>')
    body.append(f'<text class="rl" x="{LEFT-18}" y="{y-3:.0f}">{r["key"]}</text>')
    body.append(f'<text class="rs" x="{LEFT-18}" y="{y+13:.0f}">vs {r["reference"]}</text>')
    # connector, then dots on top, each with a 2px surface ring
    body.append(f'<line class="conn" x1="{tx(min(z,s)):.1f}" y1="{y:.0f}" '
                f'x2="{tx(max(z,s)):.1f}" y2="{y:.0f}"/>')
    body.append(f'<circle class="dot ref" cx="{tx(s):.1f}" cy="{y:.0f}" r="6"/>')
    body.append(f'<circle class="dot zar" cx="{tx(z):.1f}" cy="{y:.0f}" r="6"/>')
    # One direct label per row: the ratio, placed past the slower end -- unless
    # that would run into the gap panel, in which case it goes to the left of
    # the faster end instead. Measured against the panel edge rather than
    # hoped for: the random forest row collides otherwise.
    verdict = f"{ratio:.2f}× faster" if ratio >= 1 else f"{1/ratio:.2f}× slower"
    est_w = len(verdict) * 7.2
    lx, anchor = tx(max(z, s)) + 14, "start"
    if lx + est_w > GX0 - 24:
        lx, anchor = tx(min(z, s)) - 14, "end"
    body.append(f'<text class="ratio {"win" if ratio>=1 else "loss"}" x="{lx:.1f}" '
                f'y="{y+4:.0f}" text-anchor="{anchor}">{verdict}</text>')
    # gap panel
    body.append(f'<line class="gline" x1="{GX0}" y1="{y:.0f}" x2="{gx(gap):.1f}" y2="{y:.0f}"/>')
    body.append(f'<circle class="gdot {"zar" if better else "ref"}" '
                f'cx="{gx(gap):.1f}" cy="{y:.0f}" r="5"/>')
    pct = f"{gap*100:.4f}%".rstrip("0").rstrip(".") if gap >= 1e-6 else f"{gap*100:.1e}%"
    who = "zarbor ahead" if better else "reference ahead"
    body.append(f'<text class="gl" x="{gx(gap)+12:.1f}" y="{y+1:.0f}">{pct}</text>')
    body.append(f'<text class="gw" x="{gx(gap)+12:.1f}" y="{y+13:.0f}">{who}</text>')
    body.append("</g>")

    tbody.append(
        f"<tr><th>{r['key']}</th><td>{r['reference']}</td>"
        f"<td>{fmt_score(zs, r['metric'])}</td><td>{fmt_score(ss, r['metric'])}</td>"
        f"<td>{r['metric'].upper()}</td><td>{pct}</td>"
        f"<td>{'zarbor' if better else 'reference'}</td>"
        f"<td>{fmt_ms(z)}</td><td>{fmt_ms(s)}</td><td>{verdict}</td></tr>")

# axes
ticks = [200, 500, 1000, 2000, 5000]
axis = [f'<line class="grid" x1="{tx(t):.1f}" y1="{TOP-16}" x2="{tx(t):.1f}" y2="{TOP+ROWH*len(rows)}"/>'
        f'<text class="tick" x="{tx(t):.1f}" y="{TOP-24}">{fmt_ms(t)}</text>' for t in ticks]
gticks = [(1e-6, "0.0001%"), (1e-4, "0.01%"), (1e-2, "1%")]
axis += [f'<line class="grid" x1="{gx(v):.1f}" y1="{TOP-16}" x2="{gx(v):.1f}" y2="{TOP+ROWH*len(rows)}"/>'
         f'<text class="tick" x="{gx(v):.1f}" y="{TOP-24}">{lb}</text>' for v, lb in gticks]

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>zarbor vs standard implementations</title>
<body data-palette="#2a78d6,#eb6834">
<style>
  .viz-root {{ color-scheme: light;
    --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
    --text-muted:#7a7873; --grid:#e8e7e3;
    --series-1:#2a78d6; --series-2:#eb6834; --neutral:#b9b7b1; }}
  @media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .viz-root {{
    color-scheme: dark;
    --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
    --text-muted:#8e8d84; --grid:#2e2e2c;
    --series-1:#3987e5; --series-2:#d95926; --neutral:#5a5954; }} }}
  :root[data-theme="dark"] .viz-root {{ color-scheme: dark;
    --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
    --text-muted:#8e8d84; --grid:#2e2e2c;
    --series-1:#3987e5; --series-2:#d95926; --neutral:#5a5954; }}
  body {{ margin:0; background:var(--surface-1,#fcfcfb); }}
  .viz-root {{ background:var(--surface-1); color:var(--text-primary);
    font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
    padding:28px 32px 36px; max-width:{W+64}px; margin:0 auto; }}
  h1 {{ font-size:19px; margin:0 0 4px; letter-spacing:-.01em; }}
  .sub {{ color:var(--text-secondary); margin:0 0 4px; font-size:13px; }}
  .meta {{ color:var(--text-muted); margin:0 0 18px; font-size:12px; }}
  .legend {{ display:flex; gap:20px; align-items:center; margin:0 0 6px; font-size:13px;
    color:var(--text-secondary); }}
  .legend i {{ width:11px; height:11px; border-radius:50%; display:inline-block;
    margin-right:7px; vertical-align:-1px; }}
  svg {{ display:block; overflow:visible; }}
  .rl {{ fill:var(--text-primary); font-size:13.5px; font-weight:600; text-anchor:end; }}
  .rs {{ fill:var(--text-muted); font-size:11.5px; text-anchor:end; }}
  .conn {{ stroke:var(--neutral); stroke-width:2; stroke-linecap:round; }}
  .dot {{ stroke:var(--surface-1); stroke-width:2; }}
  .zar {{ fill:var(--series-1); }} .ref {{ fill:var(--series-2); }}
  .ratio {{ font-size:12.5px; font-weight:600; fill:var(--text-secondary); }}
  .ratio.loss {{ fill:var(--text-muted); }}
  .gline {{ stroke:var(--grid); stroke-width:2; stroke-linecap:round; }}
  .gdot {{ stroke:var(--surface-1); stroke-width:2; }}
  .gdot.zar {{ fill:var(--series-1); }} .gdot.ref {{ fill:var(--series-2); }}
  .gl {{ font-size:11.5px; fill:var(--text-secondary); }}
  .gw {{ font-size:10.5px; fill:var(--text-muted); }}
  .grid {{ stroke:var(--grid); stroke-width:1; }}
  .tick {{ fill:var(--text-muted); font-size:11px; text-anchor:middle; }}
  .ptitle {{ fill:var(--text-secondary); font-size:11.5px; font-weight:600;
    letter-spacing:.04em; text-transform:uppercase; }}
  .hit {{ fill:transparent; }}
  .row:hover .hit {{ fill:var(--grid); opacity:.45; }}
  table {{ border-collapse:collapse; margin-top:26px; font-size:12.5px; width:100%; }}
  th,td {{ text-align:right; padding:6px 10px; border-bottom:1px solid var(--grid);
    color:var(--text-secondary); white-space:nowrap; }}
  thead th {{ color:var(--text-muted); font-weight:600; font-size:11px;
    text-transform:uppercase; letter-spacing:.04em; }}
  tbody th {{ text-align:left; color:var(--text-primary); font-weight:600; }}
  td:nth-child(2) {{ text-align:left; }}
  .foot {{ color:var(--text-muted); font-size:11.5px; margin:14px 0 0; max-width:1000px; line-height:1.55; }}
  .foot b {{ color:var(--text-secondary); font-weight:600; }}
  details {{ margin-top:8px; }} summary {{ cursor:pointer; color:var(--text-muted);
    font-size:12px; }}
  #tip {{ position:fixed; pointer-events:none; opacity:0; transition:opacity .08s;
    background:var(--surface-1); color:var(--text-primary); border:1px solid var(--grid);
    border-radius:7px; padding:9px 11px; font-size:12px; box-shadow:0 4px 14px #0002;
    z-index:9; line-height:1.5; }}
  #tip b {{ font-weight:600; }} #tip .k {{ color:var(--text-muted); }}
</style>
<div class="viz-root">
  <h1>zarbor against the standard implementation of each model</h1>
  <p class="sub">Kaggle Playground S6E9 &mdash; 534,932 train / 133,733 validation rows, 13 features.
     Time is <b>prepare + fit + predict</b>, so binning is charged to whichever side does it.</p>
  <p class="meta">Ryzen 7 9800X3D, 16 threads &middot; median of 5 &middot;
     XGBoost 3.4.1, LightGBM 4.7.0, scikit-learn 1.9.1 &middot; CSV parsing measured separately and charged to neither</p>
  <div class="legend">
    <span><i style="background:var(--series-1)"></i>zarbor</span>
    <span><i style="background:var(--series-2)"></i>standard implementation</span>
  </div>
  <svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
       aria-label="Dumbbell chart comparing zarbor to XGBoost, LightGBM and scikit-learn on model time, with an accuracy gap panel">
    <text class="ptitle" x="{LEFT}" y="{TOP-48}">Model time (log scale)</text>
    <text class="ptitle" x="{GX0}" y="{TOP-48}">Accuracy gap &mdash; and which side is ahead</text>
    {''.join(axis)}
    {''.join(body)}
  </svg>
  <p class="foot">Every row is the same dataset and machine, so the time column is comparable down the page.
     <b>GBDT&nbsp;+&nbsp;GOSS is the one row where the gap exceeds its own noise floor</b> (1.42&times; the
     reference's max_bin envelope) &mdash; zarbor scores higher, but that is an unexplained disagreement,
     not a result. Random forest and linear regression compare deliberately different constructions,
     so their gaps are design differences rather than defects. See docs/arena.md.</p>
  <details open><summary>Table view &mdash; every number in the figure</summary>
  <table><thead><tr><th>Model</th><th>Reference</th><th>zarbor</th><th>reference</th>
  <th>Metric</th><th>Rel. gap</th><th>Closer to truth</th>
  <th>zarbor time</th><th>ref time</th><th>Speed</th></tr></thead>
  <tbody>{''.join(tbody)}</tbody></table></details>
</div>
<div id="tip"></div>
<script>
const DATA = {json.dumps([{ 'k': r['key'], 'r': r['reference'],
   'zs': r['zarbor_score'], 'ss': r['ref_score'], 'm': r['metric'],
   'zp': r['zarbor']['prepare'], 'zf': r['zarbor']['fit'], 'zd': r['zarbor']['predict'],
   'rp': r['ref']['prepare'], 'rf': r['ref']['fit'], 'rd': r['ref']['predict'],
   'zt': r['zarbor']['model_total'], 'rt': r['ref']['model_total']} for r in rows])};
const tip=document.getElementById('tip');
const ms=v=>v>=1000?(v/1000).toFixed(2)+' s':Math.round(v)+' ms';
const sc=(v,m)=>m==='rmse'&&v>100?v.toLocaleString(undefined,{{maximumFractionDigits:2}}):v.toFixed(6);
document.querySelectorAll('.row').forEach(g=>{{
  g.addEventListener('pointermove',e=>{{
    const d=DATA[+g.dataset.i];
    tip.innerHTML=`<b>${{d.k}}</b><br><span class="k">metric</span> ${{d.m.toUpperCase()}} &mdash; `+
      `zarbor <b>${{sc(d.zs,d.m)}}</b> vs ${{d.r}} <b>${{sc(d.ss,d.m)}}</b>`+
      `<br><span class="k">zarbor</span> prepare ${{ms(d.zp)}} + fit ${{ms(d.zf)}} + predict ${{ms(d.zd)}} = <b>${{ms(d.zt)}}</b>`+
      `<br><span class="k">${{d.r}}</span> prepare ${{ms(d.rp)}} + fit ${{ms(d.rf)}} + predict ${{ms(d.rd)}} = <b>${{ms(d.rt)}}</b>`;
    tip.style.opacity=1;
    const r=tip.getBoundingClientRect();
    tip.style.left=Math.min(e.clientX+16,innerWidth-r.width-12)+'px';
    tip.style.top=Math.min(e.clientY+16,innerHeight-r.height-12)+'px';
  }});
  g.addEventListener('pointerleave',()=>tip.style.opacity=0);
}});
</script>
</body></html>"""

out = ARENA / "scoreboard.html"
out.write_text(HTML)
print("wrote", out)
