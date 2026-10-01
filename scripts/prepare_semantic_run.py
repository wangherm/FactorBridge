"""Create a new immutable semantic run using the existing admitted data manifest."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from factorbridge.io import read_json,write_json,sha256,config
from factorbridge.data import audit,manifest,load_dataset
from factorbridge.benchmark import prepare
from factorbridge.pretraining import export_text
from factorbridge.semantic import bundle_for,resolve


def create(source_config,bundle_path,run_dir,prepare_data=True):
    source=read_json(source_config)
    if source['stage']!='noise_recovery':raise ValueError('Expected a Stage 1 source configuration')
    raw=read_json(source['manifest'])
    if any(e['role'] not in {'public_train','public_validation'} or e['source_kind']=='internal_real' for e in raw['datasets']):
        raise ValueError('Use public train/validation only; this experiment cannot open frozen test data')
    root=Path(run_dir).resolve()
    if root.exists():raise FileExistsError('New run directory required')
    root.mkdir(parents=True)
    c=dict(source)
    for key in ['reuse_smoke_config','pilot_only']:
        c.pop(key,None)
    settings=read_json(Path(__file__).resolve().parents[1]/'configs/semantic_settings.json')
    c.update(settings)
    c['semantic'].update(bundle=str(Path(bundle_path).resolve()),sha256=sha256(bundle_path))
    c.update(run_dir=str(root),manifest=str(Path(c['manifest']).resolve()),internal_final_test_only=True,
        defer_public_test=False,independent_dataset_cards=True)
    write_json(root/'config.json',c); config(root/'config.json')
    b=bundle_for(c); mapping=[]
    for e in manifest(c['manifest']):
        x,genes,samples,_=load_dataset(e); mapped=resolve(b,e['species'],genes)
        annotated=sum(r is not None and bool(r['terms']) for r in mapped)
        mapping.append({'dataset_id':e['dataset_id'],'species':e['species'],'samples':len(samples),'genes':len(genes),
            'identifier_mapped':sum(r is not None for r in mapped),'function_annotated':annotated,
            'coverage':annotated/len(genes),'status':'available' if annotated else 'unavailable_no_verified_mapping'})
    write_json(root/'annotation_audit.json',{'datasets':mapping,'internal_data_used':False,
        'policy':'Unmapped genes remain numeric candidates; no guessed orthology or gene names'})
    if not any(r['function_annotated'] for r in mapping):raise ValueError('No functional coverage in actual data')
    if prepare_data:
        audit(c);prepare(c);quality=export_text(c)
        write_json(root/'preparation_status.json',{'status':'prepared','label_quality':quality,
            'supervision':'existing real weak references; function annotations are input priors, never gold',
            'gpu_smoke_executed':False,'training_executed':False,'agent_executed':False})
    return root/'config.json'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-config',required=True)
    p.add_argument('--bundle',default='data/annotations/ncbi_reactome_v1/bundle.json')
    p.add_argument('--run-dir',default='runs/semantic_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args();print('SEMANTIC_CONFIG='+str(create(a.source_config,a.bundle,a.run_dir,not a.audit_only)),flush=True)
