"""Incremental detokenization parity for Qwen byte-level edge cases."""

import argparse

from transformers import AutoTokenizer

from nanovllm.incremental_detokenizer import IncrementalDetokenizer


CASES = (
    "hello world, paged attention",
    "你好，双卡 MoE 推理系统！",
    "emoji: 🚀 café naïve",
    "```python\nprint('测试')\n```",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8",
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    for text in CASES:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoder = IncrementalDetokenizer(tokenizer)
        pieces = [decoder.push(token_id) for token_id in token_ids]
        expected = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert "".join(pieces) == expected, (text, pieces, expected)
    print("[p5-detokenizer] Qwen incremental parity passed")


if __name__ == "__main__":
    main()
