"""Contact sheet of stills, for looking over a render quickly: python eval/sheet.py OUT.jpg IMG [IMG ...] [--cols 2]"""
import sys
from PIL import Image
args = [a for a in sys.argv[2:] if not a.startswith("--")]
cols = int(sys.argv[sys.argv.index("--cols") + 1]) if "--cols" in sys.argv else 2
args = [a for a in args if a != str(cols) or a.endswith(".jpg")]
ims = [Image.open(a) for a in args]
w, h = ims[0].size
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (w * cols, h * rows))
for i, im in enumerate(ims):
    sheet.paste(im, ((i % cols) * w, (i // cols) * h))
sheet.save(sys.argv[1], quality=85)
