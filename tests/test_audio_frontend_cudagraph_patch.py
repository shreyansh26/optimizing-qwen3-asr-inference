from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time
import unittest
from unittest.mock import patch

import torch


PATCH_DIR = (
    Path(__file__).resolve().parents[1]
    / "inference"
    / "vllm_qwen3_asr_optimizations"
)
sys.path.insert(0, str(PATCH_DIR))

import audio_frontend_cudagraph_patch as frontend_patch  # noqa: E402
from audio_cpu_metadata_pack_patch import (  # noqa: E402
    _build_cpu_metadata,
    run_audio_frontend_eager,
)
from audio_frontend_cudagraph_patch import (  # noqa: E402
    ENV_NAME,
    ExactShapeAudioFrontendGraphCache,
    _NATURAL_PACKED_ROWS,
    _NoopFrontendBackend,
    _PROBATION_OBSERVATIONS,
    _TAIL_PACKED_ROWS,
    _canonical_single_audio_natural_metadata,
    _make_frontend_graph_key,
    _make_frontend_graph_signature,
    _natural_feature_lengths_for_rows,
    audio_frontend_cudagraph_enabled,
)


class _FakeConv:
    def __init__(self, shape: tuple[int, int, int], scale: float) -> None:
        self.shape = shape
        self.scale = scale

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        base = inputs.float().mean(dim=tuple(range(1, inputs.ndim)), keepdim=True)
        values = torch.arange(
            self.shape[0] * self.shape[1] * self.shape[2],
            device=inputs.device,
            dtype=torch.float32,
        ).reshape(1, *self.shape)
        output = base.reshape(batch, 1, 1, 1) + values * self.scale
        return output.to(inputs.dtype)


class _FakeLinear:
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        base = inputs.float().mean(dim=-1, keepdim=True)
        columns = torch.arange(
            1024,
            device=inputs.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1024)
        return (base + columns / 1024.0).to(inputs.dtype)


class _FailCaptureBackend(_NoopFrontendBackend):
    def __init__(self) -> None:
        self.capture_calls = 0

    def capture(self, *args, **kwargs):
        self.capture_calls += 1
        raise RuntimeError("forced capture failure")


class _RejectAdmissionBackend(_NoopFrontendBackend):
    def equal(self, left: torch.Tensor, right: torch.Tensor) -> bool:
        del left, right
        return False


class _CountingFrontendBackend(_NoopFrontendBackend):
    def __init__(self) -> None:
        self.capture_calls = 0
        self.replay_calls = 0
        self.shared_resource_calls = 0
        self.execution_stream = object()
        self.graph_pool = object()
        self.capture_streams = []
        self.capture_pools = []
        self._state_lock = threading.Lock()

    def create_shared_capture_resources(self, device):
        del device
        with self._state_lock:
            self.shared_resource_calls += 1
        return self.execution_stream, self.graph_pool

    def capture(self, *args, **kwargs):
        with self._state_lock:
            self.capture_calls += 1
            self.capture_streams.append(kwargs["execution_stream"])
            self.capture_pools.append(kwargs["graph_pool"])
        return super().capture(*args, **kwargs)

    def replay(self, graph) -> None:
        self.replay_calls += 1
        super().replay(graph)


class _TrackingReplayBackend(_NoopFrontendBackend):
    def __init__(self, transaction_delay: float = 0.01) -> None:
        self.active_transactions = 0
        self.max_active_transactions = 0
        self.transaction_delay = transaction_delay
        self._state_lock = threading.Lock()

    def replay_and_clone(self, entry, padded_feature):
        with self._state_lock:
            self.active_transactions += 1
            self.max_active_transactions = max(
                self.max_active_transactions,
                self.active_transactions,
            )
        try:
            time.sleep(self.transaction_delay)
            return super().replay_and_clone(entry, padded_feature)
        finally:
            with self._state_lock:
                self.active_transactions -= 1


def _encoder() -> SimpleNamespace:
    return SimpleNamespace(
        training=False,
        conv_chunksize=1024,
        conv2d1=_FakeConv((1, 64, 50), 0.0001),
        conv2d2=_FakeConv((1, 32, 25), 0.0002),
        conv2d3=_FakeConv((1, 16, 13), 0.0003),
        conv_out=_FakeLinear(),
        positional_embedding=SimpleNamespace(
            positional_embedding=torch.arange(
                13 * 1024,
                dtype=torch.bfloat16,
            ).reshape(13, 1024)
            / 8192
        ),
    )


def _metadata(feature_length: int = 2972):
    return _natural_metadata(feature_length)


def _padded(seed: int = 0) -> torch.Tensor:
    return _natural_padded(2972, seed=seed)


def _natural_metadata(feature_length: int):
    values = _canonical_single_audio_natural_metadata(feature_length)
    if values is None:
        raise ValueError("feature length is outside the natural hotset")
    chunk_lengths, pack_metadata, cu, feature_lens, aftercnn_lens = values
    return (
        torch.tensor(chunk_lengths, dtype=torch.int64),
        torch.tensor(pack_metadata, dtype=torch.int32),
        cu,
        feature_lens,
        aftercnn_lens,
    )


def _natural_padded(feature_length: int, seed: int = 0) -> torch.Tensor:
    chunk_lengths = _natural_metadata(feature_length)[0].tolist()
    generator = torch.Generator().manual_seed(seed)
    chunks = [
        torch.randn(
            (length, 128),
            generator=generator,
            dtype=torch.bfloat16,
        )
        for length in chunk_lengths
    ]
    return torch.nn.utils.rnn.pad_sequence(
        chunks,
        batch_first=True,
    ).transpose(1, 2).unsqueeze(1)


def _single_audio_tail_case(
    feature_length: int,
    rows: int,
    *,
    seed: int,
):
    feature_lens = torch.tensor([feature_length], dtype=torch.int64)
    aftercnn_lens = torch.tensor([rows], dtype=torch.int64)
    chunk_lengths, pack_metadata, cu_seqlens = _build_cpu_metadata(
        feature_lens,
        aftercnn_lens,
        n_window=50,
        n_window_infer=800,
    )
    generator = torch.Generator().manual_seed(seed)
    chunks = [
        torch.randn(
            (int(length), 128),
            generator=generator,
            dtype=torch.bfloat16,
        )
        for length in chunk_lengths.tolist()
    ]
    padded = torch.nn.utils.rnn.pad_sequence(
        chunks,
        batch_first=True,
    ).transpose(1, 2).unsqueeze(1)
    metadata = (
        chunk_lengths,
        pack_metadata,
        tuple(cu_seqlens),
        (feature_length,),
        (rows,),
    )
    return padded, metadata


class AudioFrontendCudaGraphPatchTest(unittest.TestCase):
    def test_environment_gate_is_strict(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(audio_frontend_cudagraph_enabled())
        with patch.dict(os.environ, {ENV_NAME: "1"}, clear=True):
            self.assertTrue(audio_frontend_cudagraph_enabled())
        for value in ("", "true", "2", "-1"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {ENV_NAME: value}, clear=True):
                    with self.assertRaises(ValueError):
                        audio_frontend_cudagraph_enabled()

    def test_key_includes_full_cpu_metadata_and_padded_shape(self) -> None:
        chunk_lengths, pack_metadata, cu_seqlens, feature_lens, aftercnn_lens = (
            _metadata()
        )
        key = _make_frontend_graph_key(
            _padded(),
            chunk_lengths,
            pack_metadata,
            cu_seqlens,
            feature_lens,
            aftercnn_lens,
        )
        self.assertIsNotNone(key)
        assert key is not None
        self.assertEqual(key.padded_shape, (30, 1, 128, 100))
        self.assertEqual(key.packed_rows, 386)
        self.assertEqual(key.chunk_lengths_values, tuple([100] * 29 + [72]))
        self.assertEqual(key.pack_metadata_values, tuple(pack_metadata.tolist()))
        self.assertEqual(key.cu_seqlens_values, cu_seqlens)
        self.assertEqual(key.feature_lens_values, feature_lens)
        self.assertEqual(key.aftercnn_lens_values, aftercnn_lens)

        altered_feature_lens = (2971,)
        altered_key = _make_frontend_graph_key(
            _padded(),
            chunk_lengths,
            pack_metadata,
            cu_seqlens,
            altered_feature_lens,
            aftercnn_lens,
        )
        self.assertNotEqual(key, altered_key)

    def test_natural_inventory_has_104_exact_keys_and_14_signatures(
        self,
    ) -> None:
        keys = []
        signatures = set()
        expected_ranges = {
            377: tuple(range(2897, 2901)),
            390: tuple(range(2997, 3001)),
        }
        for rows in range(378, 390):
            first = 2901 + (rows - 378) * 8
            expected_ranges[rows] = tuple(range(first, first + 8))

        for rows in sorted(_NATURAL_PACKED_ROWS):
            self.assertEqual(
                _natural_feature_lengths_for_rows(rows),
                expected_ranges[rows],
            )
            for feature_length in expected_ranges[rows]:
                expected_metadata = _natural_metadata(feature_length)
                built_chunks, built_pack, built_cu = _build_cpu_metadata(
                    torch.tensor([feature_length], dtype=torch.int64),
                    torch.tensor([rows], dtype=torch.int64),
                    n_window=50,
                    n_window_infer=800,
                )
                self.assertTrue(torch.equal(built_chunks, expected_metadata[0]))
                self.assertTrue(torch.equal(built_pack, expected_metadata[1]))
                self.assertEqual(tuple(built_cu), expected_metadata[2])
                padded = _natural_padded(feature_length, seed=feature_length)
                key = _make_frontend_graph_key(
                    padded,
                    *expected_metadata,
                )
                self.assertIsNotNone(key)
                assert key is not None
                self.assertEqual(key.packed_rows, rows)
                self.assertEqual(
                    key.padded_shape,
                    (
                        29 if rows == 377 else 30,
                        1,
                        128,
                        100,
                    ),
                )
                self.assertEqual(key.padded_stride, (12800, 128, 1, 128))
                keys.append(key)
                signatures.add(_make_frontend_graph_signature(key))

        self.assertEqual(len(keys), 104)
        self.assertEqual(len(set(keys)), 104)
        self.assertEqual(len(signatures), 14)

    def test_natural_key_rejects_noncanonical_full_metadata(self) -> None:
        feature_length = 2972
        padded = _natural_padded(feature_length)
        metadata = list(_natural_metadata(feature_length))
        self.assertIsNotNone(_make_frontend_graph_key(padded, *metadata))

        cases = []
        altered_chunks = metadata[0].clone()
        altered_chunks[-1] -= 1
        cases.append((altered_chunks, *metadata[1:]))
        altered_pack = metadata[1].clone()
        altered_pack[-1] -= 1
        cases.append((metadata[0], altered_pack, *metadata[2:]))
        cases.append((metadata[0], metadata[1], (0, 104, 208, 311, 384), *metadata[3:]))
        cases.append((metadata[0], metadata[1], metadata[2], (2971,), metadata[4]))
        cases.append((metadata[0], metadata[1], metadata[2], metadata[3], (383,)))
        cases.append(
            (
                metadata[0],
                metadata[1],
                metadata[2],
                (800, 2172),
                (104, 280),
            )
        )
        for altered in cases:
            with self.subTest(metadata=altered[2:]):
                self.assertIsNone(_make_frontend_graph_key(padded, *altered))

    def test_21_chunk_tail_rows_are_not_admitted(self) -> None:
        self.assertEqual(_TAIL_PACKED_ROWS, frozenset())
        feature_lengths = {
            263: 2020,
            266: 2048,
            269: 2070,
            271: 2088,
        }
        for rows, feature_length in feature_lengths.items():
            with self.subTest(rows=rows, feature_length=feature_length):
                padded, metadata = _single_audio_tail_case(
                    feature_length,
                    rows,
                    seed=rows,
                )
                key = _make_frontend_graph_key(
                    padded,
                    *metadata,
                )
                self.assertIsNone(key)

        unsupported_padded, unsupported_metadata = _single_audio_tail_case(
            2016,
            262,
            seed=262,
        )
        self.assertIsNone(
            _make_frontend_graph_key(
                unsupported_padded,
                *unsupported_metadata,
            )
        )

    def test_fake_graph_replay_is_exact_for_same_shape_different_content(self) -> None:
        encoder = _encoder()
        cache = ExactShapeAudioFrontendGraphCache(
            backend=_NoopFrontendBackend(),
            warmup_iterations=1,
        )
        metadata = _metadata()
        first = _padded(1)
        second = _padded(2)

        with torch.inference_mode():
            for _ in range(_PROBATION_OBSERVATIONS):
                output_first = cache.run(
                    encoder,
                    first,
                    *metadata,
                    async_tensor_h2d=None,
                )
            output_second = cache.run(
                encoder,
                second,
                *metadata,
                async_tensor_h2d=None,
            )
            reference_second = run_audio_frontend_eager(
                encoder,
                second,
                *metadata,
                async_tensor_h2d=None,
            )

        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.admitted_key_count, 1)
        self.assertTrue(torch.equal(output_second, reference_second))
        self.assertFalse(torch.equal(output_first, output_second))
        self.assertNotEqual(output_second.data_ptr(), reference_second.data_ptr())

    def test_probation_captures_on_eighth_observation_then_replays(self) -> None:
        encoder = _encoder()
        backend = _CountingFrontendBackend()
        cache = ExactShapeAudioFrontendGraphCache(
            backend=backend,
            warmup_iterations=1,
        )
        metadata = _metadata()
        padded = _padded(9)

        with torch.inference_mode():
            for observation in range(1, _PROBATION_OBSERVATIONS):
                output = cache.run(
                    encoder,
                    padded,
                    *metadata,
                    async_tensor_h2d=None,
                )
                self.assertEqual(cache.entry_count, 0)
                self.assertEqual(cache.admitted_key_count, 0)
                self.assertEqual(cache.probation_key_count, 1)
                self.assertEqual(backend.capture_calls, 0)
                observed_key = next(iter(cache._probation_counts))
                self.assertEqual(
                    observation,
                    cache._probation_counts[observed_key],
                )
            eighth = cache.run(
                encoder,
                padded,
                *metadata,
                async_tensor_h2d=None,
            )
            replay_input = _padded(10)
            ninth = cache.run(
                encoder,
                replay_input,
                *metadata,
                async_tensor_h2d=None,
            )
            expected_eighth = run_audio_frontend_eager(
                encoder,
                padded,
                *metadata,
                async_tensor_h2d=None,
            )
            expected_ninth = run_audio_frontend_eager(
                encoder,
                replay_input,
                *metadata,
                async_tensor_h2d=None,
            )

        self.assertTrue(torch.equal(output, expected_eighth))
        self.assertTrue(torch.equal(eighth, expected_eighth))
        self.assertTrue(torch.equal(ninth, expected_ninth))
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.admitted_key_count, 1)
        self.assertEqual(cache.probation_key_count, 0)
        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(backend.replay_calls, 2)

    def test_natural_exact_keys_share_signature_only_after_own_probation(self) -> None:
        encoder = _encoder()
        backend = _CountingFrontendBackend()
        cache = ExactShapeAudioFrontendGraphCache(
            backend=backend,
            warmup_iterations=1,
        )
        first_length, *_, alias_length = _natural_feature_lengths_for_rows(387)
        first_metadata = _natural_metadata(first_length)
        alias_metadata = _natural_metadata(alias_length)
        first_padded = _natural_padded(first_length, seed=11)
        alias_padded = _natural_padded(alias_length, seed=12)

        with torch.inference_mode():
            for _ in range(_PROBATION_OBSERVATIONS):
                cache.run(
                    encoder,
                    first_padded,
                    *first_metadata,
                    async_tensor_h2d=None,
                )
            for _ in range(_PROBATION_OBSERVATIONS - 1):
                eager_alias = cache.run(
                    encoder,
                    alias_padded,
                    *alias_metadata,
                    async_tensor_h2d=None,
                )
            self.assertEqual(cache.entry_count, 1)
            self.assertEqual(cache.admitted_key_count, 1)
            admitted_alias = cache.run(
                encoder,
                alias_padded,
                *alias_metadata,
                async_tensor_h2d=None,
            )
            replay_alias_padded = _natural_padded(alias_length, seed=13)
            replay_alias = cache.run(
                encoder,
                replay_alias_padded,
                *alias_metadata,
                async_tensor_h2d=None,
            )
            expected_alias = run_audio_frontend_eager(
                encoder,
                alias_padded,
                *alias_metadata,
                async_tensor_h2d=None,
            )
            expected_replay_alias = run_audio_frontend_eager(
                encoder,
                replay_alias_padded,
                *alias_metadata,
                async_tensor_h2d=None,
            )

        self.assertTrue(torch.equal(eager_alias, expected_alias))
        self.assertTrue(torch.equal(admitted_alias, expected_alias))
        self.assertTrue(torch.equal(replay_alias, expected_replay_alias))
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.admitted_key_count, 2)
        self.assertEqual(cache.probation_key_count, 0)
        self.assertEqual(backend.capture_calls, 1)

    def test_distinct_signatures_share_one_execution_stream_and_pool(self) -> None:
        encoder = _encoder()
        backend = _CountingFrontendBackend()
        cache = ExactShapeAudioFrontendGraphCache(
            backend=backend,
            max_entries=2,
            warmup_iterations=1,
        )
        feature_lengths = tuple(
            _natural_feature_lengths_for_rows(rows)[-1]
            for rows in (377, 378)
        )

        with torch.inference_mode():
            for seed, feature_length in enumerate(feature_lengths, start=30):
                padded = _natural_padded(feature_length, seed=seed)
                metadata = _natural_metadata(feature_length)
                for _ in range(_PROBATION_OBSERVATIONS):
                    cache.run(
                        encoder,
                        padded,
                        *metadata,
                        async_tensor_h2d=None,
                    )

        self.assertEqual(cache.entry_count, 2)
        self.assertEqual(backend.capture_calls, 2)
        self.assertEqual(backend.shared_resource_calls, 1)
        self.assertEqual(
            backend.capture_streams,
            [backend.execution_stream, backend.execution_stream],
        )
        self.assertEqual(
            backend.capture_pools,
            [backend.graph_pool, backend.graph_pool],
        )
        self.assertEqual(
            {entry.execution_stream for entry in cache._entries.values()},
            {backend.execution_stream},
        )

    def test_graph_entries_and_probation_do_not_evict(self) -> None:
        encoder = _encoder()
        cache = ExactShapeAudioFrontendGraphCache(
            backend=_NoopFrontendBackend(),
            max_entries=1,
            max_probation_keys=2,
            warmup_iterations=1,
        )
        first_length = _natural_feature_lengths_for_rows(377)[-1]
        second_length = _natural_feature_lengths_for_rows(378)[-1]

        with torch.inference_mode():
            for _ in range(_PROBATION_OBSERVATIONS):
                cache.run(
                    encoder,
                    _natural_padded(first_length, seed=20),
                    *_natural_metadata(first_length),
                    async_tensor_h2d=None,
                )
            for _ in range(_PROBATION_OBSERVATIONS):
                cache.run(
                    encoder,
                    _natural_padded(second_length, seed=21),
                    *_natural_metadata(second_length),
                    async_tensor_h2d=None,
                )
            replay_input = _natural_padded(first_length, seed=22)
            replay_output = cache.run(
                encoder,
                replay_input,
                *_natural_metadata(first_length),
                async_tensor_h2d=None,
            )
            reference = run_audio_frontend_eager(
                encoder,
                replay_input,
                *_natural_metadata(first_length),
                async_tensor_h2d=None,
            )

        self.assertTrue(torch.equal(replay_output, reference))
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.admitted_key_count, 1)
        self.assertEqual(cache.probation_key_count, 0)

    def test_probation_capacity_stays_bounded_without_evicting_seen_keys(self) -> None:
        encoder = _encoder()
        cache = ExactShapeAudioFrontendGraphCache(
            backend=_NoopFrontendBackend(),
            max_probation_keys=2,
            warmup_iterations=1,
        )
        feature_lengths = (2897, 2901, 2909)
        metadata_by_rows = [_metadata(length) for length in feature_lengths]
        keys = [
            _make_frontend_graph_key(
                _natural_padded(length, seed=length),
                *metadata,
            )
            for length, metadata in zip(
                feature_lengths,
                metadata_by_rows,
                strict=True,
            )
        ]
        self.assertTrue(all(key is not None for key in keys))

        with torch.inference_mode():
            for length, metadata in zip(
                feature_lengths,
                metadata_by_rows,
                strict=True,
            ):
                cache.run(
                    encoder,
                    _natural_padded(length, seed=length),
                    *metadata,
                    async_tensor_h2d=None,
                )

        self.assertEqual(cache.probation_key_count, 2)
        assert keys[0] is not None and keys[1] is not None and keys[2] is not None
        self.assertIn(keys[0], cache._probation_counts)
        self.assertIn(keys[1], cache._probation_counts)
        self.assertNotIn(keys[2], cache._probation_counts)

    def test_cuda_backend_records_cross_stream_tensor_lifetimes(self) -> None:
        calls = []

        class _FakeStream:
            def __init__(self, name: str) -> None:
                self.name = name

            def wait_stream(self, other) -> None:
                calls.append((self.name, "wait_stream", other.name))

            def wait_event(self, event) -> None:
                calls.append((self.name, "wait_event", event.name))

        class _FakeStreamContext:
            def __init__(self, stream) -> None:
                self.stream = stream

            def __enter__(self):
                calls.append((self.stream.name, "enter"))

            def __exit__(self, exc_type, exc_value, traceback):
                del exc_type, exc_value, traceback
                calls.append((self.stream.name, "exit"))

        class _FakeTensor:
            def __init__(self, name: str) -> None:
                self.name = name
                self.device = torch.device("cuda:0")

            def copy_(self, other, *, non_blocking=False):
                calls.append(
                    (self.name, "copy", other.name, non_blocking)
                )

            def clone(self):
                calls.append((self.name, "clone"))
                return clone

            def record_stream(self, stream) -> None:
                calls.append((self.name, "record_stream", stream.name))

        class _FakeGraph:
            def replay(self) -> None:
                calls.append(("graph", "replay"))

        class _FakeEvent:
            name = "done"

            def record(self, stream) -> None:
                calls.append((self.name, "record", stream.name))

        caller_stream = _FakeStream("caller")
        replay_stream = _FakeStream("replay")
        padded_feature = _FakeTensor("input")
        static_padded_feature = _FakeTensor("static")
        graph_output = _FakeTensor("graph_output")
        clone = _FakeTensor("clone")
        entry = SimpleNamespace(
            execution_stream=replay_stream,
            static_padded_feature=static_padded_feature,
            graph=_FakeGraph(),
            output=graph_output,
            done_event=_FakeEvent(),
        )

        backend = frontend_patch._TorchCudaGraphBackend()
        with (
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=caller_stream,
            ),
            patch.object(
                torch.cuda,
                "stream",
                side_effect=_FakeStreamContext,
            ),
        ):
            actual = backend.replay_and_clone(entry, padded_feature)

        self.assertIs(actual, clone)
        self.assertEqual(
            calls,
            [
                ("replay", "wait_stream", "caller"),
                ("replay", "enter"),
                ("static", "copy", "input", True),
                ("input", "record_stream", "replay"),
                ("graph", "replay"),
                ("graph_output", "clone"),
                ("done", "record", "replay"),
                ("replay", "exit"),
                ("caller", "wait_event", "done"),
                ("clone", "record_stream", "caller"),
            ],
        )

    def test_concurrent_first_wrapper_call_attaches_one_cache(self) -> None:
        encoder = _encoder()
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
                padded_feature,
                chunk_lengths_cpu,
                pack_metadata_cpu,
                cu_seqlens_values,
                feature_lens_values,
                aftercnn_lens_values,
                *,
                async_tensor_h2d,
            ):
                del (
                    encoder,
                    chunk_lengths_cpu,
                    pack_metadata_cpu,
                    cu_seqlens_values,
                    feature_lens_values,
                    aftercnn_lens_values,
                    async_tensor_h2d,
                )
                with self.run_lock:
                    self.run_calls += 1
                return padded_feature.clone()

        inputs = [_padded(7), _padded(8)]
        metadata = _metadata()
        barrier = threading.Barrier(2)

        def invoke(padded_feature) -> torch.Tensor:
            barrier.wait()
            return frontend_patch.run_audio_frontend_cudagraph(
                encoder,
                padded_feature,
                *metadata,
                async_tensor_h2d=None,
            )

        with patch.object(
            frontend_patch,
            "ExactShapeAudioFrontendGraphCache",
            _SlowCache,
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                outputs = list(executor.map(invoke, inputs))

        attached_cache = getattr(encoder, frontend_patch._CACHE_ATTR)
        self.assertEqual(creation_count, 1)
        self.assertEqual(attached_cache.run_calls, 2)
        for padded_feature, output in zip(inputs, outputs, strict=True):
            self.assertTrue(torch.equal(padded_feature, output))

    def test_capture_failure_falls_closed_and_marks_key_rejected(self) -> None:
        encoder = _encoder()
        backend = _FailCaptureBackend()
        cache = ExactShapeAudioFrontendGraphCache(
            backend=backend,
            warmup_iterations=1,
        )
        metadata = _metadata()
        padded = _padded(3)

        with torch.inference_mode():
            for _ in range(_PROBATION_OBSERVATIONS):
                output = cache.run(
                    encoder,
                    padded,
                    *metadata,
                    async_tensor_h2d=None,
                )
            reference = run_audio_frontend_eager(
                encoder,
                padded,
                *metadata,
                async_tensor_h2d=None,
            )
            cache.run(
                encoder,
                padded,
                *metadata,
                async_tensor_h2d=None,
            )

        self.assertTrue(torch.equal(output, reference))
        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(cache.rejected_count, 1)
        self.assertEqual(backend.capture_calls, 1)

    def test_bitwise_admission_failure_falls_closed(self) -> None:
        encoder = _encoder()
        cache = ExactShapeAudioFrontendGraphCache(
            backend=_RejectAdmissionBackend(),
            warmup_iterations=1,
        )

        with torch.inference_mode():
            for _ in range(_PROBATION_OBSERVATIONS):
                output = cache.run(
                    encoder,
                    _padded(4),
                    *_metadata(),
                    async_tensor_h2d=None,
                )

        self.assertEqual(output.shape, (386, 1024))
        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(cache.rejected_count, 1)

    def test_two_thread_same_key_replay_is_serialized_and_exact(self) -> None:
        encoder = _encoder()
        backend = _TrackingReplayBackend()
        cache = ExactShapeAudioFrontendGraphCache(
            backend=backend,
            warmup_iterations=1,
        )
        metadata = _metadata()
        with torch.inference_mode():
            for _ in range(_PROBATION_OBSERVATIONS):
                cache.run(
                    encoder,
                    _padded(5),
                    *metadata,
                    async_tensor_h2d=None,
                )
        barrier = threading.Barrier(2)

        def run(seed: int) -> torch.Tensor:
            padded = _padded(seed)
            with torch.inference_mode():
                barrier.wait()
                output = cache.run(
                    encoder,
                    padded,
                    *metadata,
                    async_tensor_h2d=None,
                )
                reference = run_audio_frontend_eager(
                    encoder,
                    padded,
                    *metadata,
                    async_tensor_h2d=None,
                )
            self.assertTrue(torch.equal(output, reference))
            return output

        with ThreadPoolExecutor(max_workers=2) as executor:
            left, right = list(executor.map(run, (6, 7)))

        self.assertEqual(cache.entry_count, 1)
        self.assertFalse(torch.equal(left, right))
        self.assertEqual(backend.max_active_transactions, 1)

    def test_two_thread_cross_signature_replay_is_serialized_and_exact(self) -> None:
        encoder = _encoder()
        backend = _TrackingReplayBackend(transaction_delay=0.01)
        cache = ExactShapeAudioFrontendGraphCache(
            backend=backend,
            max_entries=2,
            warmup_iterations=1,
        )
        feature_lengths = tuple(
            _natural_feature_lengths_for_rows(rows)[-1]
            for rows in (377, 378)
        )
        metadata = tuple(
            _natural_metadata(feature_length)
            for feature_length in feature_lengths
        )
        with torch.inference_mode():
            for index, (feature_length, values) in enumerate(
                zip(feature_lengths, metadata, strict=True)
            ):
                padded = _natural_padded(feature_length, seed=40 + index)
                for _ in range(_PROBATION_OBSERVATIONS):
                    cache.run(
                        encoder,
                        padded,
                        *values,
                        async_tensor_h2d=None,
                    )

        self.assertEqual(cache.entry_count, 2)
        backend.max_active_transactions = 0
        barrier = threading.Barrier(2)

        def run(index: int) -> torch.Tensor:
            padded = _natural_padded(feature_lengths[index], seed=50 + index)
            with torch.inference_mode():
                barrier.wait()
                output = cache.run(
                    encoder,
                    padded,
                    *metadata[index],
                    async_tensor_h2d=None,
                )
                reference = run_audio_frontend_eager(
                    encoder,
                    padded,
                    *metadata[index],
                    async_tensor_h2d=None,
                )
            self.assertTrue(torch.equal(output, reference))
            return output

        with ThreadPoolExecutor(max_workers=2) as executor:
            left, right = list(executor.map(run, (0, 1)))

        self.assertFalse(torch.equal(left, right))
        self.assertNotEqual(left.data_ptr(), right.data_ptr())
        self.assertEqual(backend.max_active_transactions, 1)


if __name__ == "__main__":
    unittest.main()
