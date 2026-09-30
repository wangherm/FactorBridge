import tempfile
from pathlib import Path
import unittest

import numpy as np

from factorbridge.benchmark import label, weak_reference, simulate, prepare
from factorbridge.evaluate import baselines, evaluate, freeze
from factorbridge.io import read_json, read_jsonl, write_json
from factorbridge.pretraining import export_text
from factorbridge.public_panel import column_key, metadata_rows


class Preparation(unittest.TestCase):
    def config(self, root):
        c = read_json(Path(__file__).resolve().parents[1] / 'configs/stage1.json')
        c.update(run_dir=str(root/'run'), rank=3, card_genes=12, bootstrap_repeats=3, noise_levels=[0,1])
        return c

    def test_dense_match_does_not_confuse_sparse_support(self):
        dense = np.ones((200,1)) / np.sqrt(200)
        sparse = np.zeros_like(dense); sparse[:16,0] = 1/4
        card = {'genes':[{'slot_id':f'g{i:03d}'} for i in range(16)]}
        c = {'reference_similarity':0.6,'min_support':3}
        args = (card,np.arange(16),np.arange(200),dense[:,0],sparse,np.array(['biological']),'public_real',c)
        old,_ = label(*args)
        new,q = label(*args,matching_w=dense)
        self.assertEqual(old['decision'],'uncertain')
        self.assertEqual(new['decision'],'retain')
        self.assertEqual(len(new['supported_gene_slots']),16)
        self.assertAlmostEqual(q['reference_similarity'],1)

    def test_zero_matching_reference_remains_uncertain(self):
        c = {'reference_similarity':0.6,'min_support':3}
        answer,_ = label({'genes':[]},[],np.arange(4),np.ones(4)/2,np.ones((4,1)),
                         np.array(['biological']),'public_real',c,matching_w=np.zeros((4,1)))
        self.assertEqual(answer['decision'],'uncertain')

    def test_deferred_test_is_absent_until_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); c=self.config(root)
            c.update(manifest=str(simulate(root/'data',studies=9)),defer_public_test=True)
            prepare(c)
            lineage=read_jsonl(root/'run/prepared/private/lineage.jsonl')
            self.assertEqual({r['split'] for r in lineage},{'train','validation'})
            self.assertEqual(read_json(root/'run/prepared/index.json')['studies'],6)
            with self.assertRaises(FileNotFoundError): prepare(c,public_test=True)
            with self.assertRaises(FileNotFoundError): evaluate(c,'test')
            quality=export_text(c)
            self.assertEqual(set(quality),{'train','validation'})
            for row in read_jsonl(root/'run/sft_text/train.jsonl'):
                self.assertEqual(set(row),{'prompt','completion'})
                self.assertNotIn('matched_reference',str(row['prompt']))
            baselines(c); evaluate(c,'validation'); freeze(c,baseline_only=True)
            prepare(c,public_test=True)
            self.assertEqual({r['split'] for r in read_jsonl(root/'run/test_prepared/private/lineage.jsonl')},{'test'})
            result=evaluate(c,'test')
            self.assertTrue(result['is_independent_evaluation'])
            path=root/'data/simulation_006.csv'
            path.write_text(path.read_text()+'\n')
            from factorbridge.evaluate import check_frozen
            with self.assertRaisesRegex(ValueError,'Frozen data changed'): check_frozen(c,[])

    def test_publisher_aliases_explicit_and_unmapped_columns_fail(self):
        self.assertEqual(column_key('GSE193258','PC9_long_washout_7d_1 [RNA-seq]'),'PC9_long_wash_1')
        self.assertEqual(column_key('GSE63577','MRC_5_PD32-1'),'MRC_5_PD32_1')
        with self.assertRaises(ValueError): metadata_rows('GSE193258',['fabricated_column'],[])


if __name__ == '__main__': unittest.main()
