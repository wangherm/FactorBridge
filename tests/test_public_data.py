"""Small parser fixtures are unit-test data, never scientific experiment inputs."""
import gzip
import tempfile
from pathlib import Path
import unittest

import numpy as np

from factorbridge.public_data import convert_gse124109, download_verified
from factorbridge.benchmark import prepare
from factorbridge.evaluate import evaluate, freeze
from factorbridge.io import read_json, write_json
from factorbridge.llm import training_data


def fixtures(root):
    titles = [f'{"Control" if day == 0 else "Day"+str(day)}_{rep}'
              for day in [0, 2, 3, 4, 6, 8, 10, 12, 14, 16] for rep in [1, 2, 3]]
    soft = []
    for i, title in enumerate(titles):
        day = int(title.split('_')[0].removeprefix('Day')) if title.startswith('Day') else 0
        condition = 'cell without serum starvation' if not day else f'cell with serum starvation for {day} days'
        soft.extend([f'^SAMPLE = GSM_fixture_{i}', f'!Sample_title = {title}',
                     '!Sample_organism_ch1 = Rattus norvegicus', '!Sample_library_strategy = RNA-Seq',
                     f'!Sample_characteristics_ch1 = cell type: {condition}',
                     '!Sample_source_name_ch1 = REF cell line',
                     f'!Sample_relation = BioSample: https://www.ncbi.nlm.nih.gov/biosample/fixture_{i}'])
    # Deliberately permuted: mapping by column position would be wrong.
    titles[0], titles[1] = titles[1], titles[0]
    matrix = 'Gene\t'+'\t'.join(titles)+'\nGeneA\t'+'\t'.join(str(i) for i in range(30))+'\n'
    a, b = root/'matrix.gz', root/'soft.gz'
    a.write_bytes(gzip.compress(matrix.encode())); b.write_bytes(gzip.compress('\n'.join(soft).encode()))
    return a, b


class PublicData(unittest.TestCase):
    def test_title_join_and_explicit_fpkm_transform(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = fixtures(Path(d))
            x, genes, samples, metadata = convert_gse124109(a, b)
            self.assertEqual(samples[0], 'GSM_fixture_1')
            self.assertEqual(metadata[0]['biological_unit'], 'fixture_1')
            np.testing.assert_allclose(x[:, 0], np.log1p(np.arange(30)))
            self.assertEqual(genes.tolist(), ['GeneA'])

    def test_misaligned_samples_and_invalid_values_fail(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = fixtures(Path(d))
            original = gzip.decompress(a.read_bytes()).decode()
            for changed in [original.replace('Control_2', 'unknown'), original.replace('GeneA\t0', 'GeneA\tnan')]:
                a.write_bytes(gzip.compress(changed.encode()))
                with self.assertRaises(ValueError): convert_gse124109(a, b)

    def test_hash_mismatch_does_not_overwrite_or_fetch(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'cached.gz'; path.write_bytes(b'bad')
            with self.assertRaises(ValueError): download_verified('https://invalid.example/', path, '0'*64)
            self.assertEqual(path.read_bytes(), b'bad')

    def test_pilot_cannot_claim_independent_evaluation_or_train(self):
        c = {'pilot_only': True}
        for split in ['validation', 'test', 'internal_test']:
            with self.assertRaises(ValueError): evaluate(c, split)
        with self.assertRaises(ValueError): evaluate(c, 'train', ['qwen_frozen'])
        with self.assertRaises(ValueError): freeze(c, baseline_only=True)
        with self.assertRaises(ValueError): training_data(c, None)

    def test_formal_pipeline_still_rejects_single_study(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, b = fixtures(root)
            x, genes, samples, metadata = convert_gse124109(a, b)
            import csv
            np.savez(root/'matrix.npz', X=x, genes=genes, samples=samples)
            with (root/'metadata.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(metadata[0])); w.writeheader(); w.writerows(metadata)
            e = dict(dataset_id='parser_fixture', study_id='parser_fixture', parent_dataset='parser_fixture',
                     matrix_path='matrix.npz', metadata_path='metadata.csv', species='rat', assay='RNA-seq', resolution='bulk',
                     data_scale='log_expression', biological_unit_col='biological_unit', condition_col='condition',
                     role='public_train', source_kind='public_real')
            write_json(root/'manifest.json', {'datasets':[e]})
            c = read_json(Path(__file__).resolve().parents[1]/'configs/stage1.json')
            c.update(manifest=str(root/'manifest.json'), run_dir=str(root/'run'))
            with self.assertRaisesRegex(ValueError, 'Independent study roles'): prepare(c)


if __name__ == '__main__':
    unittest.main()
