# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small incremental detokenizer adapted from vLLM's Apache-2.0 utility."""

from __future__ import annotations


def _replace_none(tokens):
    for index, token in enumerate(tokens):
        if token is None:
            tokens[index] = ""


def _convert_prompt_ids_to_tokens(tokenizer, prompt_ids, skip_special_tokens=True):
    tokens = tokenizer.convert_ids_to_tokens(prompt_ids[-7:], skip_special_tokens=skip_special_tokens)
    if isinstance(tokens, str):
        tokens = [tokens]
    _replace_none(tokens)
    read_offset = len(tokens)
    return tokens, max(read_offset - 5, 0), read_offset


def _detokenize_incrementally(tokenizer, token_ids, prev_tokens, prefix_offset, read_offset,
                              skip_special_tokens=True):
    first = prev_tokens is None
    if first:
        prev_tokens, prefix_offset, read_offset = _convert_prompt_ids_to_tokens(
            tokenizer, token_ids[:-1], skip_special_tokens
        )
    new_tokens = tokenizer.convert_ids_to_tokens(
        [token_ids[-1]], skip_special_tokens=skip_special_tokens
    )
    if isinstance(new_tokens, str):
        new_tokens = [new_tokens]
    _replace_none(new_tokens)
    output_tokens = prev_tokens + new_tokens
    prefix_text = tokenizer.convert_tokens_to_string(
        output_tokens[prefix_offset:read_offset]
    )
    new_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:])
    if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
        return new_tokens, "", prefix_offset, read_offset
    return new_tokens, new_text[len(prefix_text):], read_offset, len(output_tokens)


class IncrementalDetokenizer:
    """Decode only a bounded token suffix while preserving UTF-8 boundaries."""

    def __init__(self, tokenizer, skip_special_tokens=True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.token_ids = []
        self.tokens = None
        self.prefix_offset = 0
        self.read_offset = 0
        self.text = ""
        self._supports_incremental = all(
            hasattr(tokenizer, name)
            for name in ("convert_ids_to_tokens", "convert_tokens_to_string")
        )

    def push(self, token_id: int) -> str:
        self.token_ids.append(token_id)
        if not self._supports_incremental:
            decoded = self.tokenizer.decode(
                self.token_ids, skip_special_tokens=self.skip_special_tokens)
            delta = decoded[len(self.text):]
            self.text = decoded
            return delta
        new_tokens, delta, self.prefix_offset, self.read_offset = (
            _detokenize_incrementally(
                self.tokenizer,
                self.token_ids,
                self.tokens,
                self.prefix_offset,
                self.read_offset,
                self.skip_special_tokens,
            )
        )
        if self.tokens is None:
            self.tokens = []
        self.tokens.extend(new_tokens)
        self.text += delta
        return delta
