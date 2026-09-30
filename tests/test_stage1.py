import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from factorbridge.io import read_json, read_jsonl, write_json
from factorbridge.data import manifest, load_dataset, local_split, assert_lineage, preprocess
from factorbridge.factors import refit, candidates, card
from factorbridge.benchmark import simulate, prepare, noise_view
from factorbridge.contracts import validate_card, validate_output, encode_completion, messages
from factorbridge.evaluate import baselines, evaluate, freeze, check_frozen, assignment, factor_metrics


def settings(root):
    c = read_json(Path(__file__).resolve().parents[1] / "configs/stage1.json")
    c.update(run_dir=str(root / "run"), rank=3, bootstrap_repeats=3, noise_levels=[0.0, 1.0], card_genes=12)
    return c


def evidence():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16, 20))
    c = {"rank": 2, "bootstrap_repeats": 3, "card_genes": 6}
    w, _, _, r, s = candidates(x, [str(i // 2) for i in range(16)], c, 2)
    return card(w[:, 0], r[:, 0], s[0], [f"gene{i}" for i in range(20)],
                {"species": "recorded", "assay": "recorded", "resolution": "unit", "data_scale": "log_expression"}, c)[0]


class TokenizerStub:
    """Only tests mask arithmetic; never called a real model smoke test."""
    eos_token = "<eos>"
    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True):
        return json.dumps(msg) + "assistant:"
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class Contracts(unittest.TestCase):
    def test_forbid_provenance_and_truth(self):
        c = evidence()
        for key in ["label", "clean_reference", "dataset_id", "split", "target_state", "test_accuracy"]:
            wrong = dict(c, **{key: "leak"})
            with self.assertRaises(ValueError): validate_card(wrong)
        self.assertNotIn("dataset_id", json.dumps(messages(c)))

    def test_invalid_slots_and_ids(self):
        c = evidence()
        o = dict(decision="retain", supported_gene_slots=["g001", "g002", "g003"], axis_flags=[], evidence_ids=[], limitations=[])
        validate_output(o, c, 3)
        for key, value in [("supported_gene_slots", ["invented"]), ("evidence_ids", ["invented"]), ("decision", "confident")]:
            with self.assertRaises(ValueError): validate_output(dict(o, **{key: value}), c, 3)

    def test_completion_mask_and_no_truncation(self):
        c = evidence()
        target = dict(decision="uncertain", supported_gene_slots=[], axis_flags=[], evidence_ids=[], limitations=[])
        x = encode_completion(TokenizerStub(), c, target, 10000, 3)
        first = next(i for i, v in enumerate(x["labels"]) if v != -100)
        self.assertGreater(first, 0)
        self.assertTrue(all(v == -100 for v in x["labels"][:first]))
        self.assertEqual(x["labels"][first:], x["input_ids"][first:])
        with self.assertRaises(ValueError): encode_completion(TokenizerStub(), c, target, 10, 3)


class Numerical(unittest.TestCase):
    def test_group_split_regions_never_separate(self):
        meta = [{"unit": f"u{i // 3}"} for i in range(36)]
        train, test = local_split(meta, "unit", 0.25, 42)
        self.assertFalse({meta[i]["unit"] for i in train} & {meta[i]["unit"] for i in test})

    def test_lineage_all_noise_views_same_split(self):
        a = dict(study_id="s", parent_dataset="d", parent_factor=["d:f"], biological_units=["s:u"], split="train")
        assert_lineage([a, a])
        for key in ["study_id", "parent_dataset", "parent_factor", "biological_units"]:
            b = dict(study_id="s2", parent_dataset="d2", parent_factor=["d2:f"], biological_units=["s2:u"], split="test")
            b[key] = a[key]
            with self.assertRaises(ValueError): assert_lineage([a, b])

    def test_refit_known_rank_one_and_holdout_centering(self):
        signal = np.arange(20.) - 10
        expected = np.array([1., -2., 3., 0.]); expected /= np.linalg.norm(expected)
        x = signal[:, None] * expected + np.array([30, 20, 10, 1])
        w, z, hz, mean = refit(x[:15], x[15:], [0, 1, 2], expected)
        self.assertGreater(abs(w @ expected), 0.999999)
        self.assertTrue(np.allclose(hz, (x[15:] - x[:15].mean(0)) @ w))
        self.assertEqual(w[3], 0)
        self.assertGreater(abs(hz.mean()), 1)

    def test_mask_is_not_zero_expression(self):
        x = np.ones((12, 10))
        noisy, measured, op = noise_view(x, "log_expression", np.arange(8), 1, 0.3, np.random.default_rng(4))
        self.assertEqual(noisy.shape, (12, 7))
        self.assertEqual(measured.sum(), 7)
        self.assertEqual(op["depth_thinning"], "not_applicable")

    def test_counts_validation(self):
        with self.assertRaises(ValueError): preprocess(np.zeros((3, 5)), "raw_counts")

    def test_matching_sign_permutation_and_unmatched(self):
        truth = np.eye(5)[:, :2]
        w = -truth[:, ::-1]
        z = np.array([[1, 2], [2, 1], [4, 3.]])
        scores = -z[:, ::-1]
        m = factor_metrics(w, z, truth, scores, True, 0.7)
        self.assertEqual(m["recovery_recall"], 1)
        self.assertAlmostEqual(m["support_f1_all_references"], 1)
        m = factor_metrics(np.zeros((5, 0)), np.zeros((3, 0)), truth, scores, True, 0.7)
        self.assertEqual(m["recovery_recall"], 0)
        self.assertEqual(assignment(np.zeros((0, 2))), [])


class EndToEnd(unittest.TestCase):
    def test_simulation_pipeline_frozen_protocol_and_numeric_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c = settings(root)
            c["manifest"] = str(simulate(root / "sim", studies=9))
            prepare(c)
            baselines(c)
            with self.assertRaises(FileNotFoundError): evaluate(c, "test")
            result = evaluate(c, "validation")
            self.assertEqual(result["not_executed_methods"], ["qwen_frozen", "qwen_finetuned"])
            with self.assertRaises(ValueError): freeze(c)
            freeze(c, baseline_only=True)
            check_frozen(c, ["stability"])
            result = evaluate(c, "test")
            self.assertEqual(result["source_kinds"], ["controlled_simulation"])
            artifacts = list((root / "run/evaluation_test/factors").glob("*.npz"))
            self.assertTrue(artifacts)
            with np.load(artifacts[0]) as a:
                self.assertEqual(a["W"].shape[0], 96)
                self.assertEqual(a["Z_heldout"].shape[1], a["W"].shape[1])
                self.assertTrue(np.isfinite(a["W"]).all())
            changed = dict(c, gene_threshold=0.6)
            with self.assertRaises(ValueError): check_frozen(changed, ["stability"])
            with self.assertRaises(ValueError): check_frozen(c, ["qwen_frozen"])
            with self.assertRaises(FileExistsError): prepare(c)

    def test_manifest_missing_files_and_study_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            path = simulate(Path(temp) / "sim", studies=9)
            obj = read_json(path)
            obj["datasets"][4]["study_id"] = obj["datasets"][0]["study_id"]
            write_json(path, obj)
            with self.assertRaises(ValueError): manifest(path)
            obj["datasets"][4]["study_id"] = "restored"
            obj["datasets"][0]["matrix_path"] = "GSE124109_NOT_DOWNLOADED.csv"
            write_json(path, obj)
            with self.assertRaises(FileNotFoundError): manifest(path)

    def test_internal_training_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = simulate(Path(temp) / "sim", studies=9)
            obj = read_json(path)
            obj["datasets"][0]["source_kind"] = "internal_real"
            write_json(path, obj)
            with self.assertRaises(ValueError): manifest(path)


if __name__ == "__main__":
    unittest.main()
