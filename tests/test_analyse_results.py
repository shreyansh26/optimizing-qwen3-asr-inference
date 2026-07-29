from __future__ import annotations

import unittest

from inference.analyse_results import PRECISIONS, precision_for_row


class AnalyseResultsPrecisionTest(unittest.TestCase):
    def test_current_names_map_to_canonical_precision_labels(self) -> None:
        cases = {
            "predictions/results_fp8_static_qk_mrope_kv_cache_fusion/x": (
                "fp8_static_qk_mrope_kv_cache_fusion"
            ),
            "predictions/results_fp8_static_audio_encoder_cudagraphs/x": (
                "fp8_static_audio_encoder_cudagraphs"
            ),
        }
        for output_root, expected in cases.items():
            with self.subTest(output_root=output_root):
                self.assertEqual(
                    precision_for_row({"output_root": output_root}),
                    expected,
                )

    def test_historical_names_map_to_new_canonical_labels(self) -> None:
        cases = {
            "predictions/results_fp8_static_qk_prefill/x": (
                "fp8_static_qk_mrope_kv_cache_fusion"
            ),
            (
                "predictions/results_fp8_static_qk_prefill_"
                "audio_prefix_suffix_cudagraph_no_tail/x"
            ): "fp8_static_audio_encoder_cudagraphs",
        }
        for output_root, expected in cases.items():
            with self.subTest(output_root=output_root):
                self.assertEqual(
                    precision_for_row({"output_root": output_root}),
                    expected,
                )

    def test_final_configuration_is_last(self) -> None:
        self.assertEqual(
            PRECISIONS[-1],
            "fp8_static_audio_encoder_cudagraphs",
        )


if __name__ == "__main__":
    unittest.main()
