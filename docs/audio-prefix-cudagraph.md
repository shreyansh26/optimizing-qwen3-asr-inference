# Natural-hotset audio prefix CUDA graphs with a shared pool

> **Historical development note.** The shared-pool design described here was
> validated and retained, but the later `M=263..273` tail expansion was not
> promoted. The final cache has 104 exact natural probation keys and 14 graph
> signatures for `F=2897..3000`, with one shared stream/pool and no tail keys.
> See the [final implementation guide](qwen3-asr-audio-length-and-graph-fast-path.md)
> and [benchmark report](audio-natural-only-cudagraph-benchmark.md).

Branch: `opt4/audio-prefix-cudagraph-natural-shared`

Base: natural-hotset prefix commit `52e81c3`.

No PyTorch, vLLM, Triton, or other dependency version changed. The opening
branch snapshot had not yet run a GPU helper or service benchmark; subsequent
sections and the final report record the completed validation.

## Workload target

The measured natural B16 workload contains 6,161 audio-tower calls. Of those,
5,698 have five cumulative attention offsets. The nine most frequent row
buckets are:

```text
M387=847  M377=752  M380=605  M386=591  M384=582
M382=581  M379=579  M385=556  M381=526
```

Those nine buckets account for 5,619 / 6,161 calls (91.2%). The bounded
contiguous family `M=377..390` uses canonical attention offsets
`(0,104,208,312,M)`. These counts are workload evidence, not performance
evidence.

## Exact prefix metadata derivation

The accepted CPU-metadata builder uses 100-frame raw chunks. Three stride-two
convolutions produce `ceil(raw_chunk_frames / 8)` valid rows per chunk. For one
natural audio with feature length `L`, therefore:

```text
raw chunks     = ceil(L / 100)
valid rows     = 13 * floor(L / 100) + ceil((L % 100) / 8)
attention cu   = (0, 104, 208, 312, valid_rows)
```

The suffix `(M, cu)` histogram does not identify a unique prefix key. Exact
source enumeration gives 104 full `PrefixGraphKey` values:

| Packed rows | Exact feature lengths | Raw chunks | Padded shape | Valid-row pack |
|---:|---:|---:|---|---|
| 377 | 2897..2900 | 29 | `(29,1,128,100)` | `(13,) * 29` |
| 378 | 2901..2908 | 30 | `(30,1,128,100)` | `(13,) * 29 + (1,)` |
| 379 | 2909..2916 | 30 | `(30,1,128,100)` | `(13,) * 29 + (2,)` |
| 380 | 2917..2924 | 30 | `(30,1,128,100)` | `(13,) * 29 + (3,)` |
| 381 | 2925..2932 | 30 | `(30,1,128,100)` | `(13,) * 29 + (4,)` |
| 382 | 2933..2940 | 30 | `(30,1,128,100)` | `(13,) * 29 + (5,)` |
| 383 | 2941..2948 | 30 | `(30,1,128,100)` | `(13,) * 29 + (6,)` |
| 384 | 2949..2956 | 30 | `(30,1,128,100)` | `(13,) * 29 + (7,)` |
| 385 | 2957..2964 | 30 | `(30,1,128,100)` | `(13,) * 29 + (8,)` |
| 386 | 2965..2972 | 30 | `(30,1,128,100)` | `(13,) * 29 + (9,)` |
| 387 | 2973..2980 | 30 | `(30,1,128,100)` | `(13,) * 29 + (10,)` |
| 388 | 2981..2988 | 30 | `(30,1,128,100)` | `(13,) * 29 + (11,)` |
| 389 | 2989..2996 | 30 | `(30,1,128,100)` | `(13,) * 29 + (12,)` |
| 390 | 2997..3000 | 30 | `(30,1,128,100)` | `(13,) * 30` |

The production `pad_sequence(...).transpose(1,2).unsqueeze(1)` layout has
stride `(12800,128,1,128)`, not the contiguous stride produced by a direct
four-dimensional `randn`. The CUDA helper reconstructs this exact layout and
zero-padded raw tail.

Every admitted runtime key still contains:

```text
padded shape and stride
packed rows, dtype, and device
every raw chunk length
every pack length and offset
every cu_seqlens boundary
the full feature_lens tuple
the full aftercnn_lens tuple
```

Natural admission additionally requires exactly one audio, an exact feature
length in 2897..3000, derived chunk/pack metadata byte-for-byte, derived
`aftercnn_lens=(M,)`, and canonical `cu=(0,104,208,312,M)`. A multi-audio
partition with the same total `M`, or any self-consistent but noncanonical
tuple, stays eager.

## Full-key admission and graph sharing

The captured callable consumes only padded input storage, pack metadata, and
total rows. The 4 or 8 full metadata keys for a given `M` consequently reduce
to one graph-static signature. The 104 exact natural keys map to 14 signatures.
The signature includes shape, stride, packed rows, dtype/device, and every pack
metadata value; it is not an `M`-only cache.

Each full metadata key has independent probation:

```text
observations 1..7 -> identical eager prefix
observation 8:
  no signature yet -> capture, compare against eager, then admit key+signature
  signature exists -> replay, compare against eager, then admit this full key
observation 9+ -> stable-input copy, graph replay, independent output clone
```

An eighth-call mismatch rejects only that full key. A graph already proven for
another exact key remains usable. Capture exceptions, unsupported layouts,
training/grad mode, nested capture, or cache capacity all fail closed to the
accepted eager prefix.

There is no graph or probation eviction. On follow-up branch
`opt6/audio-tail-rows-263-271`, capacity is 25 graph signatures: 14 natural
signatures plus 11 tail-row slots for the contiguous `M=263..273` family.
Probation retains at most 115 exact keys; once full, unseen keys remain eager
without displacing observed state. Explicit retained tensor storage is
approximately 20.7 MiB for 14 natural signatures, versus approximately
154.0 MiB for 104 separate graphs; CUDA graph private pools and convolution
workspaces are additional.

Each signature owns stable input/metadata/output buffers and an event. The cache
now creates one execution stream and one `torch.cuda.graph_pool_handle()` lazily
on the first capture, then supplies that identical stream and pool to every
signature capture. Because graph-private allocations may be reused across
captures from the shared pool, one cache-wide transaction lock covers capture,
alias-admission replay, and hot replay across all signatures. Different graphs
therefore never execute concurrently.

Every transaction enqueues `input copy -> graph replay -> output clone` in FIFO
order before releasing the lock. The clone is created before any later replay
can overwrite shared graph storage, every returned `[M,1024]` tensor owns
independent storage, and input/clone allocator lifetimes remain protected with
`record_stream`. No device-wide synchronization was added.

The parent implementation retained 5906 MiB after capturing all 14 signatures.
Reducing that graph-pool reservation is the purpose of this branch, but the
reduction remains unproven until the same GPU helper is rerun.

## Validation

CPU gate:

```bash
rtk env \
  UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
  uv run python -m unittest tests.test_audio_prefix_cudagraph_patch -v
```

The focused suite enumerates all 104 natural keys and 14 signatures, rejects
altered/multi-audio metadata, verifies observation-eight admission and shared
signature aliasing, checks no-eviction capacity behavior, asserts that distinct
signatures receive the identical stream/pool objects, and covers allocator
lifetime plus same-key and cross-signature two-thread replay serialization.

Future all-natural-row GPU gate:

```bash
rtk env \
  CUDA_VISIBLE_DEVICES=1 \
  VLLM_LOGGING_LEVEL=WARNING \
  UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
  uv run inference/vllm_static_fp8/benchmarks/bench_audio_prefix_cudagraph.py \
    --warmup 3 \
    --repeats 10 \
    --replay-checks 3 \
    --concurrency-iterations 20
```

For every `M=377..390`, the helper advances both endpoint feature-length keys
through independent probation, requires one shared signature, checks fresh
content bitwise equality and retained-clone ownership, compares the exact CUDA
kernel name/order/count of the captured graph body against eager, and reports
full eager versus input-copy + graph + output-clone timing. It then races an
admitted `M=387` key from two host threads on two CUDA streams.

Required final marker:

```text
gate=PASS_EXACT_NATURAL_AUDIO_PREFIX_CUDAGRAPH
```

### Follow-up 21-chunk tail-row gate

Branch `opt6/audio-tail-rows-263-271` adds the missing compatible rows 263,
266, 269, and 271 without changing the prefix input topology
`(21,1,128,100)`. On 2026-07-21, GPU1 passed the chained prefix-plus-suffix
helper independently for all four new rows. Each run passed bitwise equality,
observation-eight admission, changed-content replay, and two-thread/two-stream
concurrency. Copy + graph replay + clone CUDA time was 2133.536--2137.824 us,
versus 6493.632--6804.240 us for chained eager execution.

This helper result validates exact execution but does not replace an
end-to-end service benchmark.

### Parent-branch 2026-07-20 baseline

Before shared-pool changes, commit `52e81c3` passed the command above on SM90
for all 14 rows:

- both endpoint feature-length keys per row passed bitwise admission and fresh
  repeated replay, for 28 admitted exact keys on 14 graph signatures;
- captured kernel names, order, and count exactly matched the eager graph body:
  21/21 kernels for `M=377`, and 22/22 for every `M=378..390`;
- two host threads on two distinct CUDA streams passed 20 retained-output
  iterations for `M=387`;
- all rows used production stride `(12800,128,1,128)`;
- process-local memory after all 14 captures was 547.964 MiB allocated and
  5906 MiB reserved; physical GPU memory returned to 0 MiB after process exit.

The graph is a host-launch optimization, not a GPU-work optimization. Across
the 14 rows, median-of-ten CUDA time including input copy, graph replay, and
output clone ranged from 917.968 to 950.960 us. The per-row change versus full
eager was `-1.575%..+0.142%` (negative means slower), and the across-row mean
was 941.111 us versus 935.630 us eager (`+0.586%` time). Host-call enqueue time
fell from a 358.873 us mean to 117.131 us (`67.36%` lower); per-row host
speedup was `164.7%..227.7%` under the helper's `eager/replay - 1` convention.
Service latency must decide whether reduced host launch overhead outweighs the
small extra copy/clone GPU cost.

Those numbers are a comparison baseline only. They do not validate this
shared-pool branch, and no memory, exactness, concurrency, or timing improvement
is claimed here until a new GPU gate passes.
