"""Explicit numerical fixtures, never claimed as UCE training or real validation."""
import copy
import unittest
import numpy as np

from factorbridge.cross_species import pair_card, validate_relation, validate_pair_card, pair_features, pooled_vector
from factorbridge.contracts import messages, encode_completion
from test_stage1 import evidence, TokenizerStub


class CrossSpeciesContracts(unittest.TestCase):
    def setUp(self):
        self.a = evidence()
        self.b = copy.deepcopy(self.a)
        self.a["species"], self.b["species"] = "fixture_species_a", "fixture_species_b"
        self.vectors = {g["feature_id"]: np.array([1., i + 1., 0.5]) for i, g in enumerate(self.a["genes"])}
        self.orthology = [(g["feature_id"], g["feature_id"]) for g in self.a["genes"]]

    def test_signed_flip_not_biological_reversal(self):
        original = pair_card(self.a, self.b, self.orthology, self.vectors, self.vectors)
        for g in self.b["genes"]: g["signed_loading"] *= -1
        flipped = pair_card(self.a, self.b, self.orthology, self.vectors, self.vectors)
        self.assertEqual(original["protein_embedding_evidence"], flipped["protein_embedding_evidence"])
        self.assertEqual(len(pair_features(flipped)), 5)

    def test_missing_embedding_is_explicit(self):
        with self.assertRaises(ValueError): pooled_vector(self.a, {})

    def test_pair_provenance_not_in_prompt(self):
        c = pair_card(self.a, self.b, self.orthology, self.vectors, self.vectors)
        target = {"relation": "uncertain", "supported_links": [], "evidence_ids": [], "limitations": []}
        validate_relation(target, c)
        x = encode_completion(TokenizerStub(), c, target, 10000, 3)
        self.assertIn(-100, x["labels"])
        self.assertIn("protein", str(messages(c)))
        self.assertNotIn("absolute_loading_weighted_cosine", str(messages(c)))
        with self.assertRaises(ValueError): validate_pair_card(dict(c, label="shared"))
        with self.assertRaises(ValueError): validate_relation(dict(target, supported_links=["not_present"]), c)

    def test_no_platform_confounding(self):
        self.b["assay"] = "other_assay"
        with self.assertRaises(ValueError): pair_card(self.a, self.b, self.orthology, self.vectors, self.vectors)

    def test_multiple_teacher_targets_are_separate_heads(self):
        from factorbridge.distill import teacher_keys
        c = {"stage": "cross_species_identity", "distillation_weights": {"esm2": 1., "uce": 0.5, "audited_model_c": 0.2}}
        self.assertEqual(teacher_keys(c), ["esm2", "uce", "audited_model_c"])
        c["stage"] = "noise_recovery"
        self.assertEqual(teacher_keys(c), [])


if __name__ == "__main__": unittest.main()
