from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random

ROOT = Path(__file__).resolve().parent

A = ROOT / "dataset" / "Grade A"
B = ROOT / "dataset" / "Grade B"

TARGET = 1000
SEED = 42

random.seed(SEED)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

B.mkdir(parents=True, exist_ok=True)


def make_pseudo_b(im, index):
    """Create a development-only synthetic Grade-B image."""

    im = im.convert("RGB")

    # Mild colour variation
    im = ImageEnhance.Color(
        im
    ).enhance(random.uniform(0.78, 0.96))

    # Mild contrast reduction
    im = ImageEnhance.Contrast(
        im
    ).enhance(random.uniform(0.82, 0.97))

    # Mild brightness variation
    im = ImageEnhance.Brightness(
        im
    ).enhance(random.uniform(0.90, 1.03))

    # Slight sharpness variation
    im = ImageEnhance.Sharpness(
        im
    ).enhance(random.uniform(0.75, 1.0))

    # Occasional small blur
    if random.random() < 0.35:
        im = im.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.15, 0.45)
            )
        )

    # Small horizontal flip sometimes
    if random.random() < 0.5:
        im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    filename = B / f"pseudoB_{index:04d}.jpg"

    im.save(
        filename,
        "JPEG",
        quality=92
    )


files = [
    p for p in A.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTENSIONS
]

if not files:
    raise SystemExit(
        "No images found in dataset/Grade A"
    )

# Remove only previously generated pseudo-B images.
old = list(B.glob("pseudoB_*.jpg"))

for p in old:
    try:
        p.unlink()
    except Exception as e:
        print("Could not remove:", p, e)

# Randomly select source A images.
if len(files) >= TARGET:
    selected = random.sample(files, TARGET)
else:
    selected = [
        random.choice(files)
        for _ in range(TARGET)
    ]

created = 0

for index, source in enumerate(selected, start=1):
    try:
        with Image.open(source) as im:
            make_pseudo_b(im, index)

        created += 1

    except Exception as e:
        print("Skipped:", source, e)

print()
print("=" * 60)
print("Synthetic Grade-B generation complete")
print("=" * 60)
print("Grade-A source images:", len(files))
print("Synthetic Grade-B images created:", created)
print()
print("IMPORTANT:")
print("These Grade-B images are SYNTHETIC.")
print("They are for development/testing only.")
print("Do NOT use their accuracy as final real-world Grade-B accuracy.")