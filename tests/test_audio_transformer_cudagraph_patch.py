from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import torch
from torch import nn


PATCH_DIR = (
    Path(__file__).resolve().parents[1]
    / "inference"
    / "vllm_qwen3_asr_optimizations"
)
BENCHMARK_DIR = PATCH_DIR / "benchmarks"
sys.path.insert(0, str(PATCH_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

import audio_transformer_cudagraph_patch as transformer_patch  # noqa: E402
import bench_audio_transformer_cudagraph as transformer_bench  # noqa: E402
from audio_cpu_metadata_pack_patch import run_audio_transformer_eager  # noqa: E402


class _FakeLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        if max_seqlen is None:
            raise AssertionError("fake transformer requires max_seqlen")
        output = hidden_states.clone()
        boundaries = tuple(int(value) for value in cu_seqlens.tolist())
        for left, right in zip(boundaries, boundaries[1:]):
            if right < left:
                raise AssertionError("fake transformer requires ordered boundaries")
            if right > left:
                output[left:right].add_((right - left) / 1024)
        max_value = max_seqlen.to(hidden_states.dtype) / 2048
        return output + max_value


class _DoubleWidth(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat((hidden_states, hidden_states), dim=-1)


class _FakeEncoder(nn.Module):
    def __init__(self, layer_count: int = 24) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(layer_count)])
        self.ln_post = nn.Identity()
        self.proj1 = nn.Identity()
        self.act = nn.Identity()
        self.proj2 = _DoubleWidth()
        self.eval()


class _FakeGraph:
    def __init__(self, function, output: torch.Tensor) -> None:
        self.function = function
        self.output = output

    def replay(self) -> None:
        self.output.copy_(self.function())


class _FakeBackend:
    def __init__(
        self,
        *,
        capture_error: Exception | None = None,
        capture_delay: float = 0.0,
        transaction_delay: float = 0.0,
        force_mismatch: bool = False,
    ) -> None:
        self.capture_error = capture_error
        self.capture_delay = capture_delay
        self.transaction_delay = transaction_delay
        self.force_mismatch = force_mismatch
        self.capture_calls = 0
        self.replay_calls = 0
        self.equal_calls = 0
        self.active_transactions = 0
        self.max_active_transactions = 0
        self.execution_stream = object()
        self._state_lock = threading.Lock()

    def supports(self, hidden_states, cu_seqlens, max_seqlen) -> bool:
        del hidden_states, cu_seqlens, max_seqlen
        return True

    def is_current_stream_capturing(self) -> bool:
        return False

    def capture(self, function, *, device, warmup_iterations):
        del device
        with self._state_lock:
            self.capture_calls += 1
        if self.capture_delay:
            time.sleep(self.capture_delay)
        if self.capture_error is not None:
            raise self.capture_error
        for _ in range(warmup_iterations):
            function()
        output = function()
        return _FakeGraph(function, output), output, self.execution_stream

    def replay(self, graph) -> None:
        with self._state_lock:
            self.replay_calls += 1
        graph.replay()

    def replay_and_clone(
        self,
        entry,
        hidden_states,
        cu_seqlens,
        rows,
    ) -> torch.Tensor:
        with self._state_lock:
            self.active_transactions += 1
            self.max_active_transactions = max(
                self.max_active_transactions,
                self.active_transactions,
            )
            self.replay_calls += 1
        try:
            entry.static_hidden_states[:rows].copy_(hidden_states)
            entry.static_cu_seqlens[:-1].copy_(cu_seqlens)
            if self.transaction_delay:
                time.sleep(self.transaction_delay)
            entry.graph.replay()
            return entry.output[:rows].clone()
        finally:
            with self._state_lock:
                self.active_transactions -= 1

    def equal(self, left, right) -> bool:
        with self._state_lock:
            self.equal_calls += 1
        return not self.force_mismatch and torch.equal(left, right)


def _inputs(
    values: tuple[int, ...],
    *,
    fill: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = values[-1]
    hidden_states = torch.full(
        (rows, 1024),
        fill,
        dtype=torch.bfloat16,
    )
    cu_seqlens = torch.tensor(values, dtype=torch.int32)
    max_seqlen = torch.tensor(
        104,
        dtype=torch.int32,
    )
    return hidden_states, cu_seqlens, max_seqlen


def _admit_key(
    cache,
    encoder,
    inputs,
    cu_seqlens_values: tuple[int, ...],
) -> torch.Tensor:
    output = None
    for _ in range(transformer_patch._PROBATION_OBSERVATIONS):
        output = cache.run(
            encoder,
            *inputs,
            cu_seqlens_values=cu_seqlens_values,
        )
    if output is None:
        raise AssertionError("probation threshold must be positive")
    return output


class AudioTransformerCudagraphPatchTest(unittest.TestCase):
    def test_environment_gate_is_strict(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(transformer_patch.audio_transformer_cudagraph_enabled())
        with patch.dict(
            os.environ,
            {transformer_patch.ENV_NAME: "1"},
            clear=True,
        ):
            self.assertTrue(transformer_patch.audio_transformer_cudagraph_enabled())
        for value in ("", "true", "2", "-1"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ,
                    {transformer_patch.ENV_NAME: value},
                    clear=True,
                ):
                    with self.assertRaises(ValueError):
                        transformer_patch.audio_transformer_cudagraph_enabled()

    def test_canonical_hotset_maps_14_exact_keys_to_one_bucket(self) -> None:
        self.assertEqual(transformer_patch._TAIL_ROWS, frozenset())
        self.assertEqual(
            transformer_patch._NATURAL_FULL_CHUNK_ROWS,
            frozenset(range(377, 391)),
        )
        self.assertEqual(len(transformer_patch._SUPPORTED_ROWS), 14)
        self.assertEqual(transformer_patch._NATURAL_BUCKET_ROWS, 390)
        self.assertEqual(transformer_patch._MAX_CACHE_ENTRIES, 1)
        self.assertEqual(
            transformer_patch.ExactShapeAudioTransformerGraphCache()._max_entries,
            1,
        )
        self.assertEqual(len(transformer_bench._DEFAULT_CASES), 14)

        helper_values = {
            transformer_bench._cumulative_values(segments)
            for segments in transformer_bench._DEFAULT_CASES
        }
        expected_values = {
            transformer_patch._canonical_cu_seqlens_values(rows)
            for rows in transformer_patch._SUPPORTED_ROWS
        }
        self.assertEqual(helper_values, expected_values)
        cross_key_cases = transformer_bench._select_shared_bucket_cases(
            list(transformer_bench._DEFAULT_CASES)
        )
        self.assertEqual(
            tuple(sum(segments) for segments in cross_key_cases),
            (377, 390),
        )

        for rows in sorted(transformer_patch._SUPPORTED_ROWS):
            with self.subTest(rows=rows):
                values = transformer_patch._canonical_cu_seqlens_values(rows)
                self.assertIsNotNone(values)
                hidden_states, cu_seqlens, max_seqlen = _inputs(values)
                key = transformer_patch._make_transformer_graph_key(
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    values,
                )
                self.assertIsNotNone(key)
                self.assertEqual(key.rows, rows)
                self.assertEqual(key.cu_seqlens_numel, len(values))
                self.assertEqual(key.dtype, torch.bfloat16)
                self.assertEqual(key.max_seqlen_value, 104)
                bucket_key = transformer_patch._make_bucket_key(key)
                graph_values = transformer_patch._graph_cu_seqlens_values(key)
                expected_bucket = 390
                self.assertEqual(bucket_key.bucket_rows, expected_bucket)
                self.assertEqual(
                    bucket_key.graph_cu_seqlens_numel,
                    len(values) + 1,
                )
                self.assertEqual(graph_values, (*values, expected_bucket))

        natural_max = transformer_patch._make_transformer_graph_key(
            *_inputs((0, 104, 208, 312, 390)),
            (0, 104, 208, 312, 390),
        )
        self.assertIsNotNone(natural_max)
        self.assertEqual(
            transformer_patch._graph_cu_seqlens_values(natural_max),
            (0, 104, 208, 312, 390, 390),
        )

    def test_noncanonical_boundaries_fail_key_closed(self) -> None:
        tail_inputs = _inputs((0, 104, 208, 264))
        for values in (
            (0, 88, 176, 264),
            (0, 103, 208, 264),
            (0, 104, 207, 264),
            (0, 104, 208, 263),
            (0, 104, 208, 312, 264),
        ):
            with self.subTest(family="tail", values=values):
                self.assertIsNone(
                    transformer_patch._make_transformer_graph_key(
                        *tail_inputs,
                        values,
                    )
                )

        natural_inputs = _inputs((0, 104, 208, 312, 377))
        for values in (
            (0, 94, 188, 282, 377),
            (0, 104, 208, 311, 377),
            (0, 104, 208, 313, 377),
            (0, 104, 208, 377),
            (0, 103, 207, 311, 377),
        ):
            with self.subTest(family="natural", values=values):
                self.assertIsNone(
                    transformer_patch._make_transformer_graph_key(
                        *natural_inputs,
                        values,
                    )
                )

        for values in (
            (0, 104, 208, 262),
            (0, 104, 208, 274),
            (0, 104, 208, 312, 376),
            (0, 104, 208, 312, 391),
        ):
            with self.subTest(family="unsupported", values=values):
                unsupported_inputs = _inputs(values)
                self.assertIsNone(
                    transformer_patch._make_transformer_graph_key(
                        *unsupported_inputs,
                        values,
                    )
                )

        hidden_states, cu_seqlens, max_seqlen = natural_inputs
        for invalid_max_value in (103, 105):
            invalid_max = max_seqlen.clone().fill_(invalid_max_value)
            self.assertIsNone(
                transformer_patch._make_transformer_graph_key(
                    hidden_states,
                    cu_seqlens,
                    invalid_max,
                    (0, 104, 208, 312, 377),
                )
            )

        encoder = _FakeEncoder()
        backend = _FakeBackend()
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            max_entries=1,
            warmup_iterations=1,
        )
        noncanonical_inputs = _inputs((0, 88, 176, 264))
        with torch.inference_mode():
            expected = run_audio_transformer_eager(
                encoder,
                *noncanonical_inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )
            actual = cache.run(
                encoder,
                *noncanonical_inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(backend.capture_calls, 0)
        self.assertEqual(cache._observation_counts, {})

    def test_eager_transformer_runs_all_layers_and_preserves_rows(self) -> None:
        encoder = _FakeEncoder()
        hidden_states, cu_seqlens, max_seqlen = _inputs((0, 104, 208, 264))

        with torch.inference_mode():
            output = run_audio_transformer_eager(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=(0, 104, 208, 264),
            )

        self.assertEqual(output.shape, (264, 2048))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_natural_key_replays_changed_hidden_content_bitwise(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend()
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            max_entries=2,
            warmup_iterations=2,
        )
        values = (0, 104, 208, 312, 377)
        hidden_states, cu_seqlens, max_seqlen = _inputs(values)
        key = transformer_patch._make_transformer_graph_key(
            hidden_states,
            cu_seqlens,
            max_seqlen,
            values,
        )
        self.assertIsNotNone(key)
        self.assertEqual(transformer_patch._PROBATION_OBSERVATIONS, 8)

        with torch.inference_mode():
            for observation_count in range(1, 8):
                probation_output = cache.run(
                    encoder,
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    cu_seqlens_values=values,
                )
                expected_probation = run_audio_transformer_eager(
                    encoder,
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    cu_seqlens_values=values,
                )
                self.assertTrue(
                    torch.equal(probation_output, expected_probation)
                )
                self.assertEqual(backend.capture_calls, 0)
                self.assertEqual(cache.entry_count, 0)
                self.assertEqual(
                    cache._observation_counts[key],
                    observation_count,
                )

            # Observation eight alone captures and returns eager-owned output.
            first = cache.run(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=values,
            )
            replay_input = torch.full_like(hidden_states, 0.75)
            expected = run_audio_transformer_eager(
                encoder,
                replay_input,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=values,
            )
            replayed = cache.run(
                encoder,
                replay_input,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=values,
            )
            replay_pointer = replayed.data_ptr()
            replay_input_again = replay_input + 0.25
            expected_again = run_audio_transformer_eager(
                encoder,
                replay_input_again,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=values,
            )
            replayed_again = cache.run(
                encoder,
                replay_input_again,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=values,
            )

        self.assertEqual(first.shape, (377, 2048))
        self.assertTrue(torch.equal(first, expected_probation))
        self.assertFalse(torch.equal(expected, expected_again))
        self.assertTrue(torch.equal(expected, replayed))
        self.assertTrue(torch.equal(expected_again, replayed_again))
        self.assertNotEqual(replay_pointer, replayed_again.data_ptr())
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.admitted_count, 1)
        self.assertEqual(cache.rejected_count, 0)
        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(backend.replay_calls, 3)
        self.assertEqual(backend.equal_calls, 1)
        self.assertEqual(cache._observation_counts[key], 8)
        bucket_key = transformer_patch._make_bucket_key(key)
        entry = cache._entries[bucket_key]
        self.assertEqual(entry.replay_count, 2)
        self.assertEqual(entry.static_hidden_states.shape, (390, 1024))
        self.assertEqual(entry.output.shape, (390, 2048))
        self.assertEqual(
            tuple(entry.static_cu_seqlens.tolist()),
            (0, 104, 208, 312, 377, 390),
        )

    def test_probation_capture_and_replay_logs_exact_counters(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend()
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            max_entries=1,
            warmup_iterations=1,
        )
        values = (0, 104, 208, 312, 377)
        inputs = _inputs(values)

        with (
            torch.inference_mode(),
            patch.object(transformer_patch.logger, "info") as info,
            patch.object(
                transformer_patch.time,
                "perf_counter_ns",
                side_effect=(1_000_000, 3_500_000),
            ),
        ):
            _admit_key(cache, encoder, inputs, values)
            cache.run(
                encoder,
                *inputs,
                cu_seqlens_values=values,
            )

        probation_calls = [
            call
            for call in info.call_args_list
            if "probation uses eager" in call.args[0]
        ]
        self.assertEqual(len(probation_calls), 7)
        self.assertEqual(
            probation_calls[0].args[1:],
            ((0, 104, 208, 312, 377), 390, 1, 8, 0, 1, 0, 14),
        )
        self.assertEqual(
            probation_calls[-1].args[1:],
            ((0, 104, 208, 312, 377), 390, 7, 8, 0, 1, 0, 14),
        )
        capture_call = next(
            call
            for call in info.call_args_list
            if "Captured bucketed" in call.args[0]
        )
        self.assertEqual(
            capture_call.args[1:],
            ((0, 104, 208, 312, 377, 390), 390, 8, 8, 1, 1, 2.5),
        )
        admission_call = next(
            call
            for call in info.call_args_list
            if "Admitted bitwise-exact" in call.args[0]
        )
        self.assertEqual(
            admission_call.args[1:],
            (
                (0, 104, 208, 312, 377),
                (0, 104, 208, 312, 377, 390),
                390,
                8,
                8,
                1,
                14,
            ),
        )
        replay_call = next(
            call
            for call in info.call_args_list
            if "replay active" in call.args[0]
        )
        self.assertEqual(
            replay_call.args[1:],
            ((0, 104, 208, 312, 377), 390, 8, 8, 1),
        )

    def test_tail_stays_eager_while_natural_key_captures(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend()
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            max_entries=4,
            warmup_iterations=1,
        )
        first_inputs = _inputs((0, 104, 208, 264))
        second_inputs = _inputs((0, 104, 208, 312, 377))

        with torch.inference_mode():
            first = _admit_key(
                cache,
                encoder,
                first_inputs,
                (0, 104, 208, 264),
            )
            second = _admit_key(
                cache,
                encoder,
                second_inputs,
                (0, 104, 208, 312, 377),
            )

        self.assertFalse(torch.equal(first, second))
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.admitted_count, 1)
        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(cache._observation_counts, {
            next(iter(cache._admitted_keys)): 8,
        })
        self.assertEqual(
            {key.bucket_rows for key in cache._entries},
            {390},
        )

    def test_all_14_exact_keys_gate_into_one_bucket_graph(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend()
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            warmup_iterations=1,
        )

        with torch.inference_mode():
            for rows in sorted(transformer_patch._SUPPORTED_ROWS):
                values = transformer_patch._canonical_cu_seqlens_values(rows)
                self.assertIsNotNone(values)
                inputs = _inputs(values, fill=rows / 1024)
                expected = run_audio_transformer_eager(
                    encoder,
                    *inputs,
                    cu_seqlens_values=values,
                )
                actual = _admit_key(cache, encoder, inputs, values)
                self.assertTrue(torch.equal(actual, expected))

        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.admitted_count, 14)
        self.assertEqual(cache.rejected_count, 0)
        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(backend.equal_calls, 14)
        self.assertEqual(
            {key.bucket_rows for key in cache._entries},
            {390},
        )
        entries_by_rows = {
            key.bucket_rows: entry for key, entry in cache._entries.items()
        }
        self.assertEqual(
            tuple(entries_by_rows[390].static_cu_seqlens.tolist()),
            (0, 104, 208, 312, 390, 390),
        )

    def test_sequential_warmup_shape_cannot_starve_batched_key(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend()
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            max_entries=1,
            warmup_iterations=1,
        )
        sequential_inputs = _inputs((0, 377))
        batched_inputs = _inputs((0, 104, 208, 312, 377))

        with torch.inference_mode():
            sequential_output = cache.run(
                encoder,
                *sequential_inputs,
                cu_seqlens_values=(0, 377),
            )
            self.assertEqual(cache.entry_count, 0)
            self.assertEqual(cache._observation_counts, {})
            batched_output = _admit_key(
                cache,
                encoder,
                batched_inputs,
                (0, 104, 208, 312, 377),
            )

        self.assertEqual(sequential_output.shape, (377, 2048))
        self.assertEqual(batched_output.shape, (377, 2048))
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(backend.capture_calls, 1)

    def test_concurrent_first_wrapper_call_attaches_one_cache(self) -> None:
        encoder = _FakeEncoder()
        creation_count = 0
        creation_lock = threading.Lock()

        class _SlowCache:
            def __init__(self) -> None:
                nonlocal creation_count
                time.sleep(0.02)
                with creation_lock:
                    creation_count += 1
                self.run_calls = 0
                self.run_lock = threading.Lock()

            def run(
                self,
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                *,
                cu_seqlens_values,
            ) -> torch.Tensor:
                del encoder, cu_seqlens, max_seqlen, cu_seqlens_values
                with self.run_lock:
                    self.run_calls += 1
                return hidden_states.clone()

        inputs_by_thread = [
            _inputs((0, 104, 208, 264), fill=0.5),
            _inputs((0, 104, 208, 264), fill=0.75),
        ]
        barrier = threading.Barrier(2)

        def invoke(inputs) -> torch.Tensor:
            barrier.wait()
            return transformer_patch.run_audio_transformer_cudagraph(
                encoder,
                *inputs,
                cu_seqlens_values=(0, 104, 208, 264),
            )

        with patch.object(
            transformer_patch,
            "ExactShapeAudioTransformerGraphCache",
            _SlowCache,
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(invoke, inputs)
                    for inputs in inputs_by_thread
                ]
                outputs = [future.result() for future in futures]

        attached_cache = getattr(encoder, transformer_patch._CACHE_ATTR)
        self.assertEqual(creation_count, 1)
        self.assertEqual(attached_cache.run_calls, 2)
        for inputs, output in zip(inputs_by_thread, outputs, strict=True):
            self.assertTrue(torch.equal(inputs[0], output))

    def test_concurrent_cold_same_key_captures_once(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend(capture_delay=0.02)
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            max_entries=2,
            warmup_iterations=1,
        )
        values = (0, 104, 208, 312, 377)
        first_inputs = _inputs(values, fill=0.5)
        second_inputs = _inputs(values, fill=0.75)
        barrier = threading.Barrier(2)

        with torch.inference_mode():
            for _ in range(7):
                cache.run(
                    encoder,
                    *first_inputs,
                    cu_seqlens_values=values,
                )
        self.assertEqual(backend.capture_calls, 0)

        def invoke(inputs) -> torch.Tensor:
            with torch.inference_mode():
                barrier.wait()
                return cache.run(
                    encoder,
                    *inputs,
                    cu_seqlens_values=values,
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(invoke, first_inputs)
            second_future = executor.submit(invoke, second_inputs)
            first_output = first_future.result()
            second_output = second_future.result()

        with torch.inference_mode():
            first_expected = run_audio_transformer_eager(
                encoder,
                *first_inputs,
                cu_seqlens_values=values,
            )
            second_expected = run_audio_transformer_eager(
                encoder,
                *second_inputs,
                cu_seqlens_values=values,
            )
        self.assertTrue(torch.equal(first_output, first_expected))
        self.assertTrue(torch.equal(second_output, second_expected))
        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(cache.entry_count, 1)
        exact_key = next(iter(cache._admitted_keys))
        bucket_key = cache._admitted_keys[exact_key]
        self.assertEqual(cache._observation_counts[exact_key], 8)
        self.assertEqual(cache._entries[bucket_key].replay_count, 1)

    def test_alternating_cross_key_replays_keep_owned_outputs(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend()
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            warmup_iterations=1,
        )
        first_values = (0, 104, 208, 312, 377)
        second_values = (0, 104, 208, 312, 390)
        with torch.inference_mode():
            _admit_key(
                cache,
                encoder,
                _inputs(first_values),
                first_values,
            )
            _admit_key(
                cache,
                encoder,
                _inputs(second_values),
                second_values,
            )
            requests = [
                (first_values, _inputs(first_values, fill=0.5)),
                (second_values, _inputs(second_values, fill=0.75)),
                (first_values, _inputs(first_values, fill=1.0)),
                (second_values, _inputs(second_values, fill=1.25)),
            ]
            expected = [
                run_audio_transformer_eager(
                    encoder,
                    *inputs,
                    cu_seqlens_values=values,
                )
                for values, inputs in requests
            ]
            outputs = [
                cache.run(
                    encoder,
                    *inputs,
                    cu_seqlens_values=values,
                )
                for values, inputs in requests
            ]

        for output, reference in zip(outputs, expected, strict=True):
            self.assertTrue(torch.equal(output, reference))
        self.assertEqual(len({output.data_ptr() for output in outputs}), 4)
        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.admitted_count, 2)
        bucket_key = next(iter(cache._entries))
        self.assertEqual(cache._entries[bucket_key].replay_count, 4)

    def test_concurrent_hot_cross_key_transactions_are_serialized(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend(transaction_delay=0.01)
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            max_entries=2,
            warmup_iterations=1,
        )
        first_values = (0, 104, 208, 312, 377)
        second_values = (0, 104, 208, 312, 390)
        with torch.inference_mode():
            _admit_key(
                cache,
                encoder,
                _inputs(first_values, fill=0.25),
                first_values,
            )
            _admit_key(
                cache,
                encoder,
                _inputs(second_values, fill=0.375),
                second_values,
            )

        inputs_by_thread = [
            [
                _inputs(first_values, fill=fill)
                for fill in (0.5, 0.625, 0.75)
            ],
            [
                _inputs(second_values, fill=fill)
                for fill in (1.0, 1.125, 1.25)
            ],
        ]
        values_by_thread = [first_values, second_values]
        barrier = threading.Barrier(2)

        def replay_many(
            inputs_list,
            values: tuple[int, ...],
        ) -> list[torch.Tensor]:
            outputs = []
            with torch.inference_mode():
                for inputs in inputs_list:
                    barrier.wait()
                    outputs.append(
                        cache.run(
                            encoder,
                            *inputs,
                            cu_seqlens_values=values,
                        )
                    )
            return outputs

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(replay_many, inputs_list, values)
                for inputs_list, values in zip(
                    inputs_by_thread,
                    values_by_thread,
                    strict=True,
                )
            ]
            outputs_by_thread = [future.result() for future in futures]

        with torch.inference_mode():
            for inputs_list, outputs, values in zip(
                inputs_by_thread,
                outputs_by_thread,
                values_by_thread,
                strict=True,
            ):
                for inputs, output in zip(inputs_list, outputs, strict=True):
                    expected = run_audio_transformer_eager(
                        encoder,
                        *inputs,
                        cu_seqlens_values=values,
                    )
                    self.assertTrue(torch.equal(output, expected))

        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.admitted_count, 2)
        self.assertEqual(backend.max_active_transactions, 1)
        bucket_key = next(iter(cache._entries))
        self.assertEqual(bucket_key.bucket_rows, 390)
        self.assertEqual(cache._entries[bucket_key].replay_count, 6)

    def test_capture_error_is_rejected_once_then_eager(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend(capture_error=RuntimeError("capture rejected"))
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            max_entries=2,
            warmup_iterations=1,
        )
        values = (0, 104, 208, 312, 377)
        inputs = _inputs(values)

        with torch.inference_mode():
            expected = run_audio_transformer_eager(
                encoder,
                *inputs,
                cu_seqlens_values=values,
            )
            for _ in range(7):
                probation = cache.run(
                    encoder,
                    *inputs,
                    cu_seqlens_values=values,
                )
                self.assertTrue(torch.equal(probation, expected))
            first = cache.run(
                encoder,
                *inputs,
                cu_seqlens_values=values,
            )
            second = cache.run(
                encoder,
                *inputs,
                cu_seqlens_values=values,
            )

        self.assertTrue(torch.equal(first, expected))
        self.assertTrue(torch.equal(second, expected))
        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(cache.rejected_count, 1)
        key = next(iter(cache._observation_counts))
        self.assertEqual(cache._observation_counts[key], 8)

    def test_exact_key_bitwise_mismatch_is_rejected_then_stays_eager(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend(force_mismatch=True)
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            warmup_iterations=1,
        )
        values = (0, 104, 208, 312, 377)
        inputs = _inputs(values)

        with torch.inference_mode():
            expected = run_audio_transformer_eager(
                encoder,
                *inputs,
                cu_seqlens_values=values,
            )
            for _ in range(transformer_patch._PROBATION_OBSERVATIONS):
                output = cache.run(
                    encoder,
                    *inputs,
                    cu_seqlens_values=values,
                )
                self.assertTrue(torch.equal(output, expected))
            replay_calls_after_rejection = backend.replay_calls
            later = cache.run(
                encoder,
                *inputs,
                cu_seqlens_values=values,
            )

        self.assertTrue(torch.equal(later, expected))
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.admitted_count, 0)
        self.assertEqual(cache.rejected_count, 1)
        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(backend.equal_calls, 1)
        self.assertEqual(backend.replay_calls, replay_calls_after_rejection)

    def test_natural_exact_keys_share_stable_bucket(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend()
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            max_entries=1,
            warmup_iterations=1,
        )
        first_values = (0, 104, 208, 312, 377)
        first_inputs = _inputs(first_values)
        second_values = (0, 104, 208, 312, 378)
        second_inputs = _inputs(second_values)

        with torch.inference_mode():
            _admit_key(
                cache,
                encoder,
                first_inputs,
                first_values,
            )
            expected = run_audio_transformer_eager(
                encoder,
                *second_inputs,
                cu_seqlens_values=second_values,
            )
            actual = _admit_key(cache, encoder, second_inputs, second_values)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.admitted_count, 2)
        self.assertEqual(backend.capture_calls, 1)
        second_key = transformer_patch._make_transformer_graph_key(
            *second_inputs,
            second_values,
        )
        self.assertEqual(cache._observation_counts[second_key], 8)

    def test_wrong_layer_count_uses_eager_without_capture(self) -> None:
        encoder = _FakeEncoder(layer_count=23)
        backend = _FakeBackend()
        cache = transformer_patch.ExactShapeAudioTransformerGraphCache(
            backend=backend,
            max_entries=2,
            warmup_iterations=1,
        )
        values = (0, 104, 208, 312, 377)
        inputs = _inputs(values)

        with torch.inference_mode():
            output = cache.run(
                encoder,
                *inputs,
                cu_seqlens_values=values,
            )

        self.assertEqual(output.shape, (377, 2048))
        self.assertEqual(backend.capture_calls, 0)
        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(cache._observation_counts, {})

    def test_missing_metadata_prerequisites_leave_model_unimported(self) -> None:
        with patch.dict(
            os.environ,
            {transformer_patch.ENV_NAME: "1"},
            clear=True,
        ):
            self.assertFalse(
                transformer_patch.install_audio_transformer_cudagraph_patch()
            )


if __name__ == "__main__":
    unittest.main()
