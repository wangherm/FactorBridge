"""Engineering fixtures only; never presented as public biology or training success."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from factorbridge.io import write_json,sha256
from factorbridge.semantic import hypergeom_tail,bh,enrich,normal_id,enrich_card,ablate
from factorbridge.contracts import validate_card,validate_output,messages
from factorbridge.factors import card
from factorbridge.programmes import rank_scores,time_analysis,agent_tool


def fixture():
    records={f'G{i}':{'id':f'G{i}','description':f'Fixture function {i}',
                      'terms':['T1'] if i<4 else ['T2'] if i<8 else []} for i in range(12)}
    return {'schema':'factorbridge-functional-v1','sources':[{'url':'https://example.org/test-fixture',
        'sha256':'a'*64,'retrieved_utc':'fixture'}],'species':{'fixture_species':{'genes':records,
        'aliases':{f'g{i}':f'G{i}' for i in range(12)}}},'terms':{
        'T1':{'name':'Fixture one','description':'first','source':'fixture'},
        'T2':{'name':'Fixture two','description':'second','source':'fixture'}}}


class Semantics(unittest.TestCase):
    def test_tail_and_multiple_testing(self):
        expected=sum(math.comb(4,j)*math.comb(8,4-j)/math.comb(12,4) for j in range(3,5))
        self.assertAlmostEqual(hypergeom_tail(3,12,4,4),expected)
        np.testing.assert_allclose(bh([.01,.04,.03]),[.03,.04,.04])

    def test_background_aliases_and_unknowns(self):
        b=fixture();b['species']['fixture_species']['aliases']['alias']='G0'
        rows,_=enrich(['g0','alias','g1','g2','g3','g4','g5','g6','unknown'],[0,1,2,3], 'fixture_species',b)
        t=next(r for r in rows if r['term_id']=='T1')
        self.assertEqual(t['background_size'],8); self.assertEqual(t['foreground_size'],3)
        self.assertEqual(normal_id('ENSMUSG000001.5|Name'),'ENSMUSG000001')
        self.assertEqual(normal_id('A1BG_A1BG'),'A1BG')
        with self.assertRaises(ValueError):enrich(['g0'],[0],'other_species',b)

    def test_semantic_contract_and_no_target_leak(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bundle.json';write_json(path,fixture())
            c={'card_genes':4,'gene_threshold':.5,'semantic':{'bundle':str(path),'sha256':sha256(path),
                 'max_terms':2,'max_genes':6,'enrichment_fdr':.5,'description_chars':100}}
            e={'species':'fixture_species','assay':'test','resolution':'unit','data_scale':'log_expression'}
            w=np.arange(12,0,-1,dtype=float);r=np.ones(12);genes=[f'g{i}' for i in range(12)]
            old,idx=card(w,r,.8,genes,e,c)
            new,indices=enrich_card(old,idx,w,r,genes,e,c)
            validate_card(new)
            self.assertTrue(set(idx)<=set(indices));self.assertLessEqual(len(indices),6)
            for field in ['clean_reference','target','study_id','test_score']:
                bad=copy.deepcopy(new);bad['functional_evidence'][field]=1
                with self.assertRaises(ValueError):validate_card(bad)
            for mode in ['no_semantics','shuffled_text']:
                changed=ablate(new,mode);validate_card(changed)
                self.assertEqual(changed['genes'],new['genes'])
            target={'decision':'retain','supported_gene_slots':['g001','g002','g003'],
                'axis_flags':[],'evidence_ids':['e001'],'limitations':[]}
            validate_output(target,new,3)
            target['supported_gene_slots'].append('unmeasured')
            with self.assertRaises(ValueError):validate_output(target,new,3)
            self.assertIn('curated functional evidence',messages(new)[0]['content'])

    def test_tied_ranks_and_missing_programmes(self):
        genes=[{'id':str(i),'terms':['T']} for i in range(3)]
        a,coverage=rank_scores(np.ones((2,3)),genes,[{'term_id':'T','annotated_members':3},
                                                  {'term_id':'missing','annotated_members':10}])
        np.testing.assert_allclose(a[:,0],0)
        self.assertTrue(np.isnan(a[:,1]).all()); self.assertFalse(coverage[1]['available'])

    def test_future_changes_do_not_change_fitted_model(self):
        rng=np.random.default_rng(4);t=np.repeat(np.arange(8),3);units=[f'u{i}' for i in range(24)]
        a=rng.normal(size=(24,12))+t[:,None]*.1;ids=[str(i) for i in range(12)]
        first=time_analysis(a,t,units,ids)
        self.assertEqual(first['status'],'executed')
        b=a.copy();b[first['arrays']['future_mask']]+=100
        second=time_analysis(b,t,units,ids)
        for key in ['clock_beta','curve_beta','reference_scale']:
            np.testing.assert_allclose(first['arrays'][key],second['arrays'][key])
        self.assertGreater(second['forecast_mse']['reference_mean'],first['forecast_mse']['reference_mean'])
        self.assertFalse(np.any(first['arrays']['fit_mask']&first['arrays']['future_mask']))

    def test_agent_interface_cannot_open_test(self):
        with self.assertRaises(ValueError):agent_tool('programme_analysis',config={},split='internal_test')
        with self.assertRaises(ValueError):agent_tool('read_gold',config={})


if __name__=='__main__':unittest.main()
