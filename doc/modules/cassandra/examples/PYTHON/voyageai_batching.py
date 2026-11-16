#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Request-size planning for Voyage AI by MongoDB embedding requests.

This module has no third-party dependencies. It turns per-text token counts
(obtained from the SDK with ``Client.tokenize``) into groups of input indexes that each fit into one
``POST /v1/embeddings`` request:

* at most ``MAX_INPUTS_PER_REQUEST`` inputs per request, and
* at most the model's documented total token count per request.

The groups are contiguous and in input order, so concatenating the vectors that
come back for each group yields the vectors in the original input order.
"""

from typing import Dict, List, Optional, Sequence

# Maximum number of inputs in one embeddings request, for every model.
MAX_INPUTS_PER_REQUEST = 1000

# Documented total tokens per embeddings request, by model.
#
#   *-lite models:           1,000,000
#   voyage-4, voyage-3.5:      320,000
#   all other text models:     120,000
#
# None means that no total token limit is documented for the model; only
# MAX_INPUTS_PER_REQUEST applies to it. The table is explicit on purpose: a
# model that is not listed here is an error, not a guess.
VOYAGE_TOKEN_LIMITS: Dict[str, Optional[int]] = {
    # Current generation
    "voyage-4-lite": 1_000_000,
    "voyage-4": 320_000,
    "voyage-4-large": 120_000,
    "voyage-code-4": None,
    # Previous generation (still accessible)
    "voyage-3.5-lite": 1_000_000,
    "voyage-3.5": 320_000,
    "voyage-3-large": 120_000,
    "voyage-code-3": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-law-2": 120_000,
}


def token_limit_for(model: str) -> Optional[int]:
    """
    Return the total token limit per request for a model.

    Returns None when the model has no documented total token limit.

    Raises:
        ValueError: if the model is not in VOYAGE_TOKEN_LIMITS.
    """
    if model not in VOYAGE_TOKEN_LIMITS:
        raise ValueError(
            f"Unknown model {model!r}: no token limit is defined for it. "
            f"Known models: {', '.join(sorted(VOYAGE_TOKEN_LIMITS))}"
        )
    return VOYAGE_TOKEN_LIMITS[model]


def plan_batches(
    token_counts: Sequence[int],
    max_tokens: Optional[int],
    max_inputs: int = MAX_INPUTS_PER_REQUEST,
) -> List[List[int]]:
    """
    Group input indexes into request-sized batches, preserving input order.

    Args:
        token_counts: Token count of each input, in input order.
        max_tokens: Total tokens allowed in one request, or None for no limit.
        max_inputs: Inputs allowed in one request.

    Returns:
        Lists of indexes. Each index appears exactly once, the groups follow
        input order, and no group exceeds max_inputs inputs or max_tokens
        total tokens. Empty input gives an empty list.

    Raises:
        ValueError: if a single input is larger than max_tokens (it could never
            be sent), if max_inputs is below 1, or if a count is negative.
    """
    if max_inputs < 1:
        raise ValueError(f"max_inputs must be at least 1, got {max_inputs}")

    batches: List[List[int]] = []
    current: List[int] = []
    current_tokens = 0

    for index, n_tokens in enumerate(token_counts):
        if n_tokens < 0:
            raise ValueError(f"Negative token count {n_tokens} at index {index}")
        if max_tokens is not None and n_tokens > max_tokens:
            raise ValueError(
                f"Input {index} has {n_tokens:,} tokens, above the "
                f"{max_tokens:,} tokens allowed in one request"
            )

        too_many_inputs = len(current) >= max_inputs
        too_many_tokens = max_tokens is not None and current_tokens + n_tokens > max_tokens
        if current and (too_many_inputs or too_many_tokens):
            batches.append(current)
            current = []
            current_tokens = 0

        current.append(index)
        current_tokens += n_tokens

    if current:
        batches.append(current)

    return batches
