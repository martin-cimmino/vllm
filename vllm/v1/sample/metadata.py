# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors


@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    no_penalties: bool
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: list[list[int]]

    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: torch.Tensor | None

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]

    # Loaded logits processors
    logitsprocs: LogitsProcessors

    # Speculative token ids
    spec_token_ids: list[list[int]] | None = None

    # Per-request SMC alpha values (0.0 means SMC disabled for that request).
    smc_alphas: torch.Tensor | None = None
    # Per-request α ramp durations (0 means no ramp, i.e. full α from step 0).
    smc_alpha_ramp_tokens: torch.Tensor | None = None
    # Per-request decode step counts (for α ramp computation).
    smc_step_counts: torch.Tensor | None = None
