import argparse
import os
import re
from typing import Any

import datasets
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from verl.utils.dataset.vision_utils import process_image
from verl.utils.reward_score.jigsaw import compute_score


def build_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert prompt strings containing <image>/<video> into Qwen-style segment lists."""
    ex = dict(example)
    messages = ex.pop("prompt")

    for message in messages:
        content = message["content"]
        content_list: list[dict[str, Any]] = []
        segments = re.split(r"(<image>|<video>)", content)
        segments = [s for s in segments if s]
        for segment in segments:
            if segment == "<image>":
                content_list.append({"type": "image"})
            elif segment == "<video>":
                content_list.append({"type": "video"})
            else:
                content_list.append({"type": "text", "text": segment})
        message["content"] = content_list

    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline inference on jigsaw_small using Qwen3-VL.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HF model id or local path.",
    )
    parser.add_argument(
        "--parquet",
        default="our_datasets/dermogpt/jigsaw_small/test.parquet",
        help="Parquet file path (relative to repo root or absolute).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max examples (0 = all).")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--device-map", default="auto", help="HF device_map (e.g. auto, balanced).")
    args = parser.parse_args()

    # Keep consistent with training pipeline defaults.
    os.environ.setdefault("QWEN3_VL_MIN_PIXELS", str(args.min_pixels))
    os.environ.setdefault("QWEN3_VL_MAX_PIXELS", str(args.max_pixels))

    parquet_path = args.parquet
    if not os.path.isabs(parquet_path):
        parquet_path = os.path.join(os.getcwd(), parquet_path)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    ds = datasets.load_dataset("parquet", data_files=parquet_path)["train"]
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    total = 0
    score_sum = 0.0

    # For device_map models, inputs should live on the first parameter device.
    input_device = next(model.parameters()).device

    for i, ex in enumerate(ds):
        messages = build_messages(ex)
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        pil_images = [process_image(img) for img in ex.get("images", [])]

        proc_kwargs = {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels}
        inputs = processor(
            text=[prompt_text],
            images=pil_images if pil_images else None,
            return_tensors="pt",
            **proc_kwargs,
        )
        inputs = {k: (v.to(input_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        decoded = processor.batch_decode(out, skip_special_tokens=False)[0]
        gt = (ex.get("extra_info") or {}).get("answer", "")
        s = compute_score(decoded, gt)

        total += 1
        score_sum += float(s)

        print(f"\n--- Example {i} ---")
        print(f"GT: {gt}")
        print(f"Score: {s:.3f}")
        print(decoded)

    if total > 0:
        print(f"\n=== Summary ===")
        print(f"Examples: {total}")
        print(f"Mean score: {score_sum / total:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

