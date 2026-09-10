from pathlib import Path
from PIL import Image
from collections import Counter

for cls in ["Grade A", "Grade B", "Grade C"]:
    folder = Path("dataset") / cls

    if not folder.exists():
        print(f"{cls}: folder missing")
        continue

    files = [
        f for f in folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]

    sizes = []
    bad = []

    for f in files:
        try:
            with Image.open(f) as im:
                im.verify()

            with Image.open(f) as im:
                sizes.append(im.size)

        except Exception:
            bad.append(f.name)

    print()
    print("=" * 50)
    print(cls)
    print("Images:", len(files))
    print("Bad/unreadable:", len(bad))
    print("Common image sizes:")

    for size, count in Counter(sizes).most_common(10):
        print(" ", size, "->", count)

    if bad:
        print("Bad files:")
        for name in bad[:20]:
            print(" ", name)

print()
print("Dataset audit complete.")