import argparse
import io
import os
from typing import Any

import pandas as pd
from PIL import Image


def _to_pil(img_data: Any) -> Image.Image | None:
    if hasattr(img_data, "save"):
        return img_data
    if isinstance(img_data, dict) and "bytes" in img_data:
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    if isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Quickly inspect a parquet row and optionally export images.")
    parser.add_argument("path", help="Path to a parquet file.")
    parser.add_argument("--row", type=int, default=0, help="Row index to inspect.")
    parser.add_argument("--image_key", default="images", help="Column name containing images.")
    parser.add_argument("--out_dir", default="inspect_parquet", help="Directory to write exported images.")
    args = parser.parse_args()

    print(f"Reading {args.path}...")
    df = pd.read_parquet(args.path)
    print(f"Total rows: {len(df)}")
    if len(df) == 0:
        return

    row = df.iloc[args.row]
    print("Columns:", list(df.columns))

    # Print a couple common keys if present.
    for k in ("data_source", "id", "ability"):
        if k in df.columns:
            print(f"{k}: {row[k]}")

    if "prompt" in df.columns:
        try:
            prompt_preview = str(row["prompt"])[:300]
            print(f"prompt preview: {prompt_preview}...")
        except Exception:
            pass

    if args.image_key not in df.columns or row.get(args.image_key, None) is None:
        print(f"No '{args.image_key}' column or empty value; done.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    images = row[args.image_key]
    if not isinstance(images, (list, tuple)):
        images = [images]

    exported = 0
    for i, img_data in enumerate(images):
        img = _to_pil(img_data)
        if img is None:
            print(f"Skipping unknown image type at index {i}: {type(img_data)}")
            continue
        out_path = os.path.join(args.out_dir, f"row{args.row}_img{i}.png")
        img.save(out_path)
        exported += 1
        print(f"Saved {out_path}")

    print(f"Exported {exported} image(s).")


if __name__ == "__main__":
    main()
