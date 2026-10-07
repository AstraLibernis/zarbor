#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 AstraLibernis

"""Render bench/arena/scoreboard.html to fig_light.png, fig_dark.png and fig_hover.png.

    python bench/arena/scoreboard.py && zig build tools -- figure && python bench/arena/render_figure.py

Needs playwright's chromium. On Fedora its shared libraries (nspr, nss, ...) may need
installing first.
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

ARENA = Path(__file__).resolve().parent
URL = (ARENA / "scoreboard.html").as_uri()
VIEW = {"width": 1200, "height": 860}

with sync_playwright() as p:
    b = p.chromium.launch()
    for theme in ("light", "dark"):
        pg = b.new_page(viewport=VIEW, device_scale_factor=2, color_scheme=theme)
        pg.goto(URL)
        pg.evaluate(f"document.documentElement.setAttribute('data-theme', '{theme}')")
        pg.wait_for_timeout(300)
        pg.locator("svg").first.screenshot(path=str(ARENA / f"fig_{theme}.png"))
        if theme == "light":
            # The tooltip, over the first CatBoost row.
            box = pg.locator("g.row").nth(5).bounding_box()
            pg.mouse.move(box["x"] + 600, box["y"] + box["height"] / 2)
            pg.wait_for_timeout(300)
            pg.screenshot(path=str(ARENA / "fig_hover.png"), clip={"x": 0, "y": 0, **VIEW})
        pg.close()
    b.close()
print("wrote fig_light.png, fig_dark.png, fig_hover.png")
