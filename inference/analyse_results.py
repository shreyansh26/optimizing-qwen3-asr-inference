import argparse
import csv
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent / "results"
PRECISIONS = (
    "bf16",
    "fp8_dynamic",
    "fp8_static",
    "fp8_static_qk_mrope_kv_cache_fusion",
    "fp8_static_audio_encoder_cudagraphs",
)
FULL_COMPLETED_COUNT = 550
PARTIAL_AUDIO_LENGTH = 50.0

FULL_COLUMNS = [
    ("latency_p50_s", "lat p50"),
    ("latency_p95_s", "lat p95"),
    ("latency_p99_s", "lat p99"),
    ("ttft_p50_s", "ttft p50"),
    ("ttft_p95_s", "ttft p95"),
    ("ttft_p99_s", "ttft p99"),
    ("throughput_files_s", "throughput"),
    ("cer", "cer"),
    ("wer", "wer"),
]

PARTIAL_COLUMNS = [
    ("latency_p50_s", "lat p50"),
    ("latency_p95_s", "lat p95"),
    ("latency_p99_s", "lat p99"),
    ("ttft_p50_s", "ttft p50"),
    ("ttft_p95_s", "ttft p95"),
    ("ttft_p99_s", "ttft p99"),
    ("throughput_files_s", "throughput"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("batched", "sequential"), required=True)
    return parser.parse_args()


def is_float(value: str, target: float) -> bool:
    try:
        return float(value) == target
    except ValueError:
        return False


def is_int(value: str, target: int) -> bool:
    try:
        return int(value) == target
    except ValueError:
        return False


def precision_for_row(row: dict[str, str]) -> str | None:
    output_root = row.get("output_root", "").lower()
    if "fp8_dynamic" in output_root:
        return "fp8_dynamic"
    if "fp8_static_audio_encoder_cudagraphs" in output_root:
        return "fp8_static_audio_encoder_cudagraphs"
    if "fp8_static_qk_mrope_kv_cache_fusion" in output_root:
        return "fp8_static_qk_mrope_kv_cache_fusion"
    if "fp8_static" in output_root:
        return "fp8_static"
    if "bf16" in output_root:
        return "bf16"
    return None


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def newest_by_precision(
    rows: list[dict[str, str]],
    predicate,
) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        precision = precision_for_row(row)
        if precision is None or not predicate(row):
            continue
        existing = selected.get(precision)
        if existing is None or row.get("timestamp_utc", "") > existing.get(
            "timestamp_utc", ""
        ):
            selected[precision] = row
    return selected


def precision_sort_value(row: dict[str, str]) -> int:
    precision = precision_for_row(row)
    if precision in PRECISIONS:
        return PRECISIONS.index(precision)
    return len(PRECISIONS)


def worker_sort_value(row: dict[str, str]) -> int:
    try:
        return int(row.get("workers", ""))
    except ValueError:
        return -1


def matching_rows(
    rows: list[dict[str, str]],
    predicate,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if precision_for_row(row) is not None and predicate(row)
    ]
    return sorted(
        selected,
        key=lambda row: (
            precision_sort_value(row),
            worker_sort_value(row),
            row.get("timestamp_utc", ""),
        ),
    )


def format_value(value: str) -> str:
    if value == "":
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return value


def sequential_table_rows(
    selected: dict[str, dict[str, str]],
    columns: list[tuple[str, str]],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for precision in PRECISIONS:
        row = selected.get(precision)
        if row is None:
            rows.append([precision.upper()] + ["n/a"] * len(columns))
            continue
        rows.append(
            [precision.upper()]
            + [format_value(row.get(column_name, "")) for column_name, _ in columns]
        )
    return rows


def batched_table_rows(
    selected: list[dict[str, str]],
    columns: list[tuple[str, str]],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in selected:
        precision = precision_for_row(row)
        rows.append(
            [
                "n/a" if precision is None else precision.upper(),
                row.get("workers", "") or "n/a",
            ]
            + [format_value(row.get(column_name, "")) for column_name, _ in columns]
        )
    return rows


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print(title)
        print("(no matching rows)")
        print()
        return

    widths = [
        max(len(row[index]) for row in [headers] + rows)
        for index in range(len(headers))
    ]
    print(title)
    print(
        " | ".join(
            header.ljust(widths[index])
            for index, header in enumerate(headers)
        )
    )
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print()


def main() -> None:
    args = parse_args()
    csv_path = RESULTS_DIR / f"{args.mode}.csv"
    rows = read_rows(csv_path)

    full_predicate = lambda row: (
        is_int(row.get("completed", ""), FULL_COMPLETED_COUNT)
        and row.get("uniform_audio_length", "") == ""
    )
    partial_predicate = lambda row: is_float(
        row.get("uniform_audio_length", ""),
        PARTIAL_AUDIO_LENGTH,
    )

    print(f"Source: {csv_path}")
    if args.mode == "batched":
        full_rows = matching_rows(rows, full_predicate)
        partial_rows = matching_rows(rows, partial_predicate)
        full_headers = ["precision", "workers"] + [header for _, header in FULL_COLUMNS]
        partial_headers = (
            ["precision", "workers"] + [header for _, header in PARTIAL_COLUMNS]
        )
        full_table_rows = batched_table_rows(full_rows, FULL_COLUMNS)
        partial_table_rows = batched_table_rows(partial_rows, PARTIAL_COLUMNS)
    else:
        full_rows = newest_by_precision(rows, full_predicate)
        partial_rows = newest_by_precision(rows, partial_predicate)
        full_headers = ["precision"] + [header for _, header in FULL_COLUMNS]
        partial_headers = ["precision"] + [header for _, header in PARTIAL_COLUMNS]
        full_table_rows = sequential_table_rows(full_rows, FULL_COLUMNS)
        partial_table_rows = sequential_table_rows(partial_rows, PARTIAL_COLUMNS)

    print_table(
        f"Full benchmark ({FULL_COMPLETED_COUNT} measured files)",
        full_headers,
        full_table_rows,
    )
    print_table(
        f"{PARTIAL_AUDIO_LENGTH:g}s audio-limit benchmark",
        partial_headers,
        partial_table_rows,
    )


if __name__ == "__main__":
    main()
