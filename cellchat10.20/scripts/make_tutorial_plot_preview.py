"""Create a labeled visual-QA contact sheet from completed tutorial figures."""
from pathlib import Path
import json

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "tutorials" / "development"
folders = sorted(p for p in RUNS.glob("[0-9][0-9]_*") if (p / "summary.json").is_file())
assert len(folders) == 7, "The preview is generated only after all seven tutorial executions"
width, height = 600, 360
canvas = Image.new("RGB", (3 * width, len(folders) * height), "#eeeeee")
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
records = []
for row, folder in enumerate(folders):
    assert json.loads((folder / "summary.json").read_text())["status"] == "passed"
    figures = sorted((folder / "tutorial").rglob("*.png"))
    # Sample beginning, middle and end of the original plotting sequence.
    selected = [figures[0], figures[len(figures) // 2], figures[-1]]
    for col, figure in enumerate(selected):
        x, y = col * width, row * height
        with Image.open(figure) as original:
            thumb = ImageOps.contain(original.convert("RGB"), (width - 16, height - 54))
            canvas.paste(thumb, (x + (width - thumb.width) // 2, y + 46 + (height - 54 - thumb.height) // 2))
        draw.text((x + 8, y + 5), folder.name, font=font, fill="black")
        draw.text((x + 8, y + 25), figure.name[:70], font=font, fill="black")
        records.append({"tutorial": folder.name, "figure": str(figure), "row": row, "column": col})
output = ROOT / "validation" / "tutorial_plot_montage.png"
canvas.save(output)
(output.with_suffix(".json")).write_text(json.dumps({"scope": "21 representative figures from the seven completed candidate tutorials", "figures": records}, indent=2))
print(output)
