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

"""Failure-mode tests for voyageai_batching. Run: python3 -m unittest test_voyageai_batching"""

import unittest

from voyageai_batching import (
    MAX_INPUTS_PER_REQUEST,
    VOYAGE_TOKEN_LIMITS,
    plan_batches,
    token_limit_for,
)


class PlanBatchesTest(unittest.TestCase):
    def test_empty_input_gives_no_batches(self):
        self.assertEqual(plan_batches([], 120_000), [])

    def test_input_count_limit_splits_small_texts(self):
        # 2,500 one-token texts are far below any token limit, but a request
        # takes at most 1000 inputs.
        batches = plan_batches([1] * 2_500, 1_000_000)
        self.assertEqual([len(b) for b in batches], [1000, 1000, 500])

    def test_input_count_limit_applies_without_token_limit(self):
        batches = plan_batches([1] * 1_001, None)
        self.assertEqual([len(b) for b in batches], [MAX_INPUTS_PER_REQUEST, 1])

    def test_token_limit_splits_before_it_is_exceeded(self):
        batches = plan_batches([60, 60, 60], 120)
        self.assertEqual(batches, [[0, 1], [2]])

    def test_batch_may_hit_the_token_limit_exactly(self):
        self.assertEqual(plan_batches([60, 60], 120), [[0, 1]])

    def test_order_is_kept_and_every_index_appears_once(self):
        counts = [5, 90, 40, 70, 1, 119, 3, 3, 3]
        batches = plan_batches(counts, 120, max_inputs=3)
        self.assertEqual([i for batch in batches for i in batch], list(range(len(counts))))
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            self.assertLessEqual(sum(counts[i] for i in batch), 120)

    def test_single_text_above_token_limit_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "Input 1 has 121 tokens"):
            plan_batches([10, 121, 10], 120)

    def test_negative_count_is_an_error(self):
        with self.assertRaises(ValueError):
            plan_batches([1, -1], 120)

    def test_max_inputs_below_one_is_an_error(self):
        with self.assertRaises(ValueError):
            plan_batches([1], 120, max_inputs=0)


class TokenLimitForTest(unittest.TestCase):
    def test_unknown_model_is_an_error_not_a_default(self):
        with self.assertRaisesRegex(ValueError, "Unknown model 'voyage-nope'"):
            token_limit_for("voyage-nope")

    def test_documented_limits(self):
        self.assertEqual(token_limit_for("voyage-4-lite"), 1_000_000)
        self.assertEqual(token_limit_for("voyage-3.5"), 320_000)
        self.assertEqual(token_limit_for("voyage-law-2"), 120_000)

    def test_code_4_has_no_documented_token_limit(self):
        self.assertIsNone(token_limit_for("voyage-code-4"))

    def test_every_listed_model_has_limit_or_none(self):
        for model in VOYAGE_TOKEN_LIMITS:
            limit = token_limit_for(model)
            self.assertTrue(limit is None or limit > 0, model)


if __name__ == "__main__":
    unittest.main()
