"""Build the app: embeds data/artifacts.json into the template.

Outputs
  frontend/index.html   standalone PWA served by the backend (full HTML document)
  dist/hekaya-artifact.html  body-only version for the claude.ai artifact demo
Run: python frontend/build.py
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
kb = json.loads((root / "data" / "artifacts.json").read_text(encoding="utf-8"))
tpl = (root / "frontend" / "app.template.html").read_text(encoding="utf-8")
body = tpl.replace("/*__KB__*/null", json.dumps(kb, ensure_ascii=False))

(root / "dist").mkdir(exist_ok=True)
(root / "dist" / "hekaya-artifact.html").write_text(body, encoding="utf-8")

head, rest = body.split("</style>", 1)
page = ('<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<link rel="icon" href="icon.svg"><link rel="apple-touch-icon" href="icon.svg">'
        '<style>:root{padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom)}'
        'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>'
        + head + "</style></head><body>" + rest + "</body></html>")
(root / "frontend" / "index.html").write_text(page, encoding="utf-8")
print("built", len(page) // 1024, "KB")
