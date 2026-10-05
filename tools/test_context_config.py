"""Tests for tools/context_config.py, the run-<model>.sh [CONTEXT] argument: the context parser, and the flags it
redoes (--max-context, rope past 262,144, --kv, KV streaming only where setup would keep it).  No GPU, no downloads.

    python -m unittest tools.test_context_config
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_config as cc  # noqa: E402

SRC = Path("strata-iq2_xs.json")                       # IQ2_XS: ram_gb 48 in setup.py MODELS
BASE = ["--pack", "p", "--max-context", "32768", "--kv", "int8"]


def run(args, ctx, ram=128.0, wsl=False):
    with mock.patch.object(cc, "mem_gb", return_value=ram), mock.patch.object(cc, "is_wsl", return_value=wsl):
        return cc.with_context(list(args), ctx, SRC)


class Parse(unittest.TestCase):
    def test_suffixes(self):
        self.assertEqual(cc.parse_context("131072"), 131072)
        self.assertEqual(cc.parse_context("512K"), 524288)
        self.assertEqual(cc.parse_context("1m"), 1048576)

    def test_refused(self):
        for bad in ("", "1.5K7", "abc", "100", "0.3K"):
            with self.assertRaises(ValueError):
                cc.parse_context(bad)


class Flags(unittest.TestCase):
    def test_rope_only_past_the_trained_context(self):
        self.assertIsNone(cc.value(run(BASE, 262144), "--rope-scaling"))
        a = run(BASE, 1048576)
        self.assertEqual((cc.value(a, "--rope-scaling"), cc.value(a, "--rope-scale")), ("yarn", "4"))

    def test_an_explicit_linear_method_is_kept_and_its_factor_derived_again(self):
        a = run(BASE + ["--rope-scaling", "linear", "--rope-scale", "3"], 524288)
        self.assertEqual((cc.value(a, "--rope-scaling"), cc.value(a, "--rope-scale")), ("linear", "2"))

    def test_kv_int8_added_above_8k_only_when_missing(self):
        self.assertIsNone(cc.value(run(["--max-context", "8192"], 8192), "--kv"))
        self.assertEqual(cc.value(run(["--max-context", "8192"], 32768), "--kv"), "int8")
        self.assertEqual(cc.value(run(BASE[:-2] + ["--kv", "q4_0"], 131072), "--kv"), "q4_0")

    def test_streaming_from_a_short_source_follows_the_ram_check(self):
        self.assertIn("--kv-resident", run(BASE, 131072, ram=64))       # 48 + 1.8 + 1 fits
        self.assertNotIn("--kv-resident", run(BASE, 1048576, ram=60))   # 48 + 14.4 + 1 does not
        self.assertNotIn("--kv-resident", run(BASE, 32768))             # under 64K: never

    def test_streaming_from_a_long_source_keeps_setups_choice(self):
        on = ["--max-context", "131072", "--kv", "int8", "--kv-resident", "32768"]
        off = ["--max-context", "131072", "--kv", "int8"]
        self.assertIn("--kv-resident", run(on, 524288, ram=16))
        self.assertNotIn("--kv-resident", run(off, 524288, ram=512))

    def test_never_streams_with_k8v4_or_on_wsl(self):
        self.assertNotIn("--kv-resident", run(BASE[:-2] + ["--kv", "k8v4"], 131072))
        self.assertNotIn("--kv-resident", run(BASE, 131072, wsl=True))

    def test_family_tag_is_stripped_for_the_model_lookup(self):
        self.assertEqual(cc.model_ram_gb(Path("strata-coder-iq3_xxs.json")), cc.model_ram_gb(Path("strata-iq3_xxs.json")))


if __name__ == "__main__":
    unittest.main()
