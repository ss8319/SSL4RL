import argparse
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small parquet subset for quick sanity runs.")
    parser.add_argument(
        "--input_dir",
        default="our_datasets/dermogpt/jigsaw",
        help="Directory containing train/valid/test parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        default="our_datasets/dermogpt/jigsaw_small",
        help="Directory to write the small subset parquet files.",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of rows to keep per split.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for split in ["train", "valid", "test"]:
        input_path = os.path.join(args.input_dir, f"{split}.parquet")
        output_path = os.path.join(args.output_dir, f"{split}.parquet")

        print(f"Processing {split} split...")
        df = pd.read_parquet(input_path)
        small_df = df.head(args.n)
        small_df.to_parquet(output_path)
        print(f"Saved {len(small_df)} examples to {output_path}")


if __name__ == "__main__":
    main()
