import json
from pathlib import Path

out = Path(__file__).resolve().parent / "train.jsonl"
rows = [
    {"instruction":"Classify the buying priority for a portable analytics laptop.","input":"16GB RAM, long battery life, lightweight, €899", "output":"PORTABLE_ANALYTICS"},
    {"instruction":"Classify the buying priority for a business development laptop.","input":"upgradeable memory, durable keyboard, €949", "output":"BUSINESS_DEV"},
    {"instruction":"Classify the buying priority for a GPU-heavy engineering laptop.","input":"discrete GPU, 16-inch display, €1399", "output":"GPU_ENGINEERING"},
]
with out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(out)
