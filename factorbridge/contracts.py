"""A strict, small visible schema. IDs and target data never enter prompt text."""
import json
import math

CARD_KEYS = {"task", "species", "assay", "resolution", "data_scale", "genes", "pathway_evidence", "regulon_evidence", "qc_associations", "replicate_evidence", "available_evidence_ids", "missing_evidence"}
GENE_KEYS = {"slot_id", "feature_id", "loading_rank", "signed_loading", "noisy_view_recurrence", "measured"}
OUTPUT_KEYS = {"decision", "supported_gene_slots", "axis_flags", "evidence_ids", "limitations"}
SYSTEM = ('Select only evidence-supported measured gene slots for numerical factor recovery. '
          'Return one JSON object with exactly decision (retain, uncertain, reject_null), '
          'supported_gene_slots, axis_flags (biological, composition, technical, other_biology), '
          'evidence_ids, limitations. Use only supplied slots and evidence IDs. '
          'Allow uncertainty. Stability alone does not establish biological origin. '
          'Do not invent genes, pathway names, probabilities, or explanations outside JSON.')


def validate_card(c):
    if c.get("task") == "cross_species_identity":
        from .cross_species import validate_pair_card
        return validate_pair_card(c)
    if set(c) != CARD_KEYS or c["task"] != "recover_noisy_factor":
        raise ValueError("Unexpected card fields: possible provenance/label leakage")
    for k in ("species", "assay", "resolution", "data_scale"):
        if not isinstance(c[k], str) or not c[k]:
            raise ValueError("Invalid card context")
    slots, genes = set(), set()
    for g in c["genes"]:
        if set(g) != GENE_KEYS or g["measured"] is not True:
            raise ValueError("Only explicitly measured gene records accepted")
        if g["slot_id"] in slots or g["feature_id"] in genes:
            raise ValueError("Duplicate slots/genes")
        slots.add(g["slot_id"])
        genes.add(g["feature_id"])
        if not math.isfinite(g["signed_loading"]) or not 0 <= g["noisy_view_recurrence"] <= 1:
            raise ValueError("Invalid noisy statistics")
    # v0.1 has no external annotation implementation: reject arbitrary nested content.
    if any(c[k] for k in ("pathway_evidence", "regulon_evidence", "qc_associations")):
        raise ValueError("External evidence not implemented in v0.1")
    if len(c["replicate_evidence"]) != 1 or set(c["replicate_evidence"][0]) != {"evidence_id", "bootstrap_cosine"}:
        raise ValueError("Unexpected replicate evidence schema")
    if c["available_evidence_ids"] != [c["replicate_evidence"][0]["evidence_id"]]:
        raise ValueError("Evidence ID mismatch")
    if not 0 <= c["replicate_evidence"][0]["bootstrap_cosine"] <= 1.00001:
        raise ValueError("Invalid stability")
    allowed_missing = {"external_annotation", "measured_technical_covariates", "composition_information"}
    if set(c["missing_evidence"]) != allowed_missing:
        raise ValueError("Unexpected missing_evidence")


def validate_output(o, c, min_support):
    if c.get("task") == "cross_species_identity":
        from .cross_species import validate_relation
        return validate_relation(o, c)
    validate_card(c)
    if not isinstance(o, dict) or set(o) != OUTPUT_KEYS:
        raise ValueError("Invalid output schema")
    if o["decision"] not in {"retain", "uncertain", "reject_null"}:
        raise ValueError("Invalid decision")
    for key in OUTPUT_KEYS - {"decision"}:
        if not isinstance(o[key], list) or any(not isinstance(v, str) for v in o[key]) or len(o[key]) != len(set(o[key])):
            raise ValueError(f"Invalid list: {key}")
    if not set(o["supported_gene_slots"]) <= {g["slot_id"] for g in c["genes"]}:
        raise ValueError("Invalid gene slot")
    if not set(o["evidence_ids"]) <= set(c["available_evidence_ids"]):
        raise ValueError("Invalid evidence ID")
    if not set(o["axis_flags"]) <= {"biological", "composition", "technical", "other_biology"}:
        raise ValueError("Invalid axis flag")
    if o["decision"] == "retain" and len(o["supported_gene_slots"]) < min_support:
        raise ValueError("Retain has insufficient support")
    return o


def messages(c):
    validate_card(c)
    system = SYSTEM
    if c.get("task") == "cross_species_identity":
        from .cross_species import PAIR_SYSTEM
        system = PAIR_SYSTEM
        # The teacher's numeric score is an auxiliary target, never a copyable input.
        c = dict(c, protein_embedding_evidence={k: v for k, v in c["protein_embedding_evidence"].items() if k != "absolute_loading_weighted_cosine"})
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(c, ensure_ascii=False, separators=(",", ":"))}]


def encode_completion(tokenizer, c, target, max_length, min_support):
    validate_output(target, c, min_support)
    prefix = tokenizer.apply_chat_template(messages(c), tokenize=False, add_generation_prompt=True)
    completion = json.dumps(target, ensure_ascii=False, separators=(",", ":")) + tokenizer.eos_token
    # Encoding separately keeps exact prompt masking and prevents truncation into answer.
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = tokenizer.encode(completion, add_special_tokens=False)
    ids = prompt_ids + answer_ids
    if len(ids) > max_length:
        raise ValueError(f"Example too long: {len(ids)} > {max_length}; reduce card_genes and re-prepare explicitly")
    if not answer_ids:
        raise ValueError("No completion tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prompt_ids) + answer_ids}
