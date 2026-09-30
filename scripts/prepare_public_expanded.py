"""Convert pinned public files and prepare an expanded SFT run; no tests or smoke.

Reuse the project's converters, grouped split, evidence cards and weak supervision.
Only recipes with reviewed sample mapping are admitted. Acquisition is separate.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import io
from pathlib import Path
import re
import sys
import tarfile

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factorbridge.io import read_json, write_json, sha256, digest
from factorbridge.public_data import download_verified, soft_samples, fetch_gse124109
from factorbridge.public_panel import fetch_public_panel, characteristics
from factorbridge.data import manifest, audit
from factorbridge.benchmark import prepare, supervision_audit
from factorbridge.pretraining import export_text


def matrix_text(path, delimiter, offset=1, gene_col=0):
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        rows = csv.reader(stream, delimiter=delimiter)
        header = next(rows); genes, values = [], []
        for row in rows:
            if len(row) != len(header): raise ValueError(f'Malformed matrix row in {path}')
            genes.append(row[gene_col]); values.append(row[offset:])
    return np.asarray(values, dtype=float).T, np.asarray(genes), header[offset:]


def base_row(s, unit, condition, key):
    bios = [x.rsplit('/', 1)[-1] for x in s.get('relation', []) if x.startswith('BioSample: ')]
    if len(bios) != 1: raise ValueError('One deposited BioSample required')
    return {'sample_id': s['accession'][0], 'biological_unit': unit, 'biosample': bios[0],
            'global_unit_id': key+':'+unit, 'condition': condition, 'sample_title': s['title'][0]}


def archive_matrix(path, samples, acc):
    columns, genes, values = [], None, []
    by_id = {s['accession'][0]: s for s in samples}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not member.isfile(): continue
            gsm = re.match(r'(GSM\d+)_', Path(member.name).name)
            if not gsm or gsm[1] not in by_id: raise ValueError('Archive entry cannot be joined to GEO sample')
            with gzip.open(archive.extractfile(member), 'rt', encoding='utf-8-sig') as f:
                rows = list(csv.reader(f, delimiter=',' if acc=='GSE167665' else '\t'))
            header = rows[0]; data = rows[1:]
            expected = 'expression' if acc=='GSE167665' else 'TPM'
            idx = header.index(expected)
            current = [r[0] for r in data]
            if genes is not None and genes != current: raise ValueError('Archive samples have different gene order/set')
            genes = current; values.append([float(r[idx]) for r in data]); columns.append(gsm[1])
    if set(columns) != set(by_id) or len(set(columns)) != len(columns):
        raise ValueError('Incomplete/duplicate archive sample mapping')
    return np.asarray(values), np.asarray(genes), columns


def convert(spec, folder):
    acc = spec['accession']; samples = soft_samples(folder/(acc+'_family.soft.gz'))
    ch = {s['accession'][0]: characteristics(s) for s in samples}
    path = folder/spec['matrix']; rows=[]
    if acc in {'GSE167665', 'GSE255958'}:
        x, genes, columns = archive_matrix(path, samples, acc)
        lookup = {s['accession'][0]: s for s in samples}
        for key in columns:
            s=lookup[key]; c=ch[key]
            if acc=='GSE167665':
                unit=c['individual']; condition='age='+c['age']+';tissue='+c['tissue']
            else:
                rep=re.search(r'rep\s*(\d+)',s['title'][0],re.I)
                # Pair the same cell-line replicate across every perturbation.
                # Organoid/PDX replicates stay grouped by the named patient model.
                line=c.get('cell line')
                if line and line!='n/a':
                    unit=line+(':rep_'+rep[1] if rep else ':all_replicates')
                else:
                    model=re.match(r'(TH107|TH021|LG0812)\b',s['title'][0])
                    if not model:raise ValueError('Unmapped patient model')
                    unit=model[1]
                condition=c.get('treatment',s['title'][0])
            rows.append(base_row(s,unit,condition,acc))
    elif acc=='GSE132040':
        x,genes,columns=matrix_text(path,',')
        summaries={'__no_feature','__ambiguous','__too_low_aQual','__not_aligned','__alignment_not_unique'}
        found={str(g) for g in genes if str(g).startswith('__')}
        if found!=summaries:raise ValueError('Unexpected HTSeq bookkeeping rows')
        keep=np.array([g not in summaries for g in genes])
        x=x[:,keep];genes=genes[keep]
        write_json(folder/'excluded_htseq_rows.json',{'excluded_non_gene_rows':sorted(summaries),
                   'reason':'HTSeq mapping bookkeeping, not gene expression; excluded before library-size normalization'})
        lookup={re.search(r'\[([^\]]+)\]$',s['title'][0])[1]:s for s in samples}
        for key in columns:
            s=lookup[key.removesuffix('.gencode.vM19')]; c=ch[s['accession'][0]]
            # The deposited source-name suffix is NOT a verified mouse ID.
            # Pool all same-age/sex animals conservatively so none can cross roles.
            unit=c['age']+':'+c['sex']
            rows.append(base_row(s,unit,'tissue='+c['tissue']+';age='+c['age'],acc))
            rows[-1]['tissue']=c['tissue']
    elif acc=='GSE189073':
        matrices=[]; all_columns=[]; gene_arrays=[]
        for name in spec['matrices']:
            a,g,cols=matrix_text(folder/name,',')
            matrices.append(a);gene_arrays.append(g);all_columns.extend(cols)
        if spec.get('gene_join')!='intersection_all_subexperiments':raise ValueError('Explicit gene intersection recipe required')
        common=set.intersection(*(set(g) for g in gene_arrays))
        genes=np.array([g for g in gene_arrays[0] if g in common])
        if len(genes)<100:raise ValueError('Insufficient common measured genes')
        aligned=[]
        for a,g in zip(matrices,gene_arrays):
            indices={name:i for i,name in enumerate(g)}
            aligned.append(a[:,[indices[name] for name in genes]])
        write_json(folder/'gene_intersection.json',{'retained':len(genes),'no_zero_imputation':True,
                   'excluded_by_file':{name:sorted(set(g)-common) for name,g in zip(spec['matrices'],gene_arrays)}})
        x=np.concatenate(aligned); columns=all_columns
        lookup={s['title'][0].replace(', RNA Rep','_').replace('uninjured','0hpi'):s for s in samples}
        for key in columns:
            s=lookup[key]; c=ch[s['accession'][0]]
            # Same condition cohort stays together; no invented animal IDs.
            rows.append(base_row(s,key.rsplit('_',1)[0],c['treatment']+';time='+c['time'],acc))
    else:
        raise ValueError(f'No admitted converter for {acc}')
    if not np.isfinite(x).all() or np.any(x<0): raise ValueError(f'{acc}: nonfinite/negative expression; no imputation')
    if len(set(genes))!=len(genes) or len(rows)!=x.shape[0]: raise ValueError('Duplicate genes or invalid sample alignment')
    if spec['transform']=='log1p_cpm':
        totals=x.sum(axis=1,keepdims=True)
        if np.any(totals<=0): raise ValueError('Empty library')
        x=np.log1p(1e6*x/totals)
    elif spec['transform']=='log1p': x=np.log1p(x)
    elif spec['transform']!='identity': raise ValueError('Unknown transformation')
    return x,genes,rows


def build_data(root, panel_dir, catalogue_dir, recipes):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    target=root/'manifest.json'
    if target.exists():
        saved=read_json(root/'conversion_report.json')
        if saved['recipes_sha256']!=sha256(recipes): raise ValueError('Recipes changed; use a new data destination')
        for path,expected in saved['derived_sha256'].items():
            if sha256(root/path)!=expected: raise ValueError(f'Converted artifact changed: {path}')
        manifest(target); return target
    if (root/'BUILD_STARTED.json').exists(): raise FileExistsError('Incomplete conversion: inspect it and use a new destination')
    write_json(root/'BUILD_STARTED.json',{'recipes_sha256':sha256(recipes)})
    entries=manifest(fetch_public_panel(panel_dir))
    for e in entries:
        # Explicit user scope revision: previously deferred public test becomes
        # training in this new protocol; it may never be reported as unseen test.
        if e['role']=='public_test': e['role']='public_train'
        if e['dataset_id']=='GSE63577': e['parent_dataset']='JenAge_fibroblast_family'
        e.pop('n_units',None)
    entries.extend(manifest(fetch_gse124109(Path(catalogue_dir)/'GSE124109_pilot')))
    report=[]
    for spec in read_json(recipes)['studies']:
        acc=spec['accession']; folder=Path(catalogue_dir).resolve()/acc
        print('ACQUIRE/CONVERT',acc,flush=True)
        for resource in spec['resources']:
            download_verified(resource['url'],folder/resource['filename'],resource['sha256'])
        x,genes,rows=convert(spec,folder)
        # Tissue panels share the parent study and global cohort IDs. They are
        # not counted as new independent studies or split across model roles.
        excluded=[]
        if acc=='GSE132040':
            excluded=[r for r in rows if r['tissue']=='NA']
            groups=sorted({r['tissue'] for r in rows if r['tissue']!='NA'})
            write_json(root/'GSE132040_excluded_samples.json',{'reason':'No tissue, age or sex assigned; no biological unit can be established','samples':excluded})
        else:groups=[None]
        for group in groups:
            ix=[i for i,r in enumerate(rows) if group is None or r['tissue']==group]
            key=acc+('_'+re.sub(r'[^A-Za-z0-9]+','_',group) if group else '')
            dest=root/key; dest.mkdir(parents=True)
            selected=[rows[i] for i in ix]
            np.savez_compressed(dest/'matrix.npz',X=x[ix],genes=genes,samples=np.array([r['sample_id'] for r in selected]))
            with (dest/'metadata.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
            entry={'dataset_id':key,'study_id':acc,'parent_dataset':spec.get('parent',acc),
                   'matrix_path':str(dest/'matrix.npz'),'metadata_path':str(dest/'metadata.csv'),
                   'species':spec['species'],'assay':'bulk_RNA_seq','resolution':'bulk_population',
                   'data_scale':'log_expression','biological_unit_col':'biological_unit','condition_col':'condition',
                   'role':spec['role'],'source_kind':'public_real'}
            entries.append(entry)
            report.append({'dataset_id':key,'samples':len(ix),'unit_groups':len({r['biological_unit'] for r in selected}),
                           'unit_policy':spec['unit_policy'],'limitations':spec['limitations'],'transform':spec['transform']})
    # Public BioSample reuse may not cross roles, regardless of accession aliases.
    owners={}
    for e in entries:
        with Path(e['metadata_path']).open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                bio=row.get('biosample')
                if bio and bio in owners and owners[bio]!=e['role']: raise ValueError(f'BioSample leakage: {bio}')
                if bio:owners[bio]=e['role']
    write_json(target,{'datasets':entries,'internal_killifish':'absent_frozen_final_test',
                      'protocol':'Public train/validation only; GSE113957 reassigned before test evaluation by explicit user request'})
    validated=manifest(target)
    derived={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}
    write_json(root/'conversion_report.json',{'recipes_sha256':sha256(recipes),'derived_sha256':derived,
        'new_datasets':report,'studies':len({e['study_id'] for e in validated}),
        'roles':{role:sorted({e['study_id'] for e in validated if e['role']==role}) for role in ['public_train','public_validation']},
        'all_catalogue_studies_admitted':False,'internal_data_used':False})
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-config',required=True,help='Original passed GPU smoke config; preserve its run')
    p.add_argument('--panel-dir',default='data/public/pretraining_panel_v1')
    p.add_argument('--catalogue-dir',default='data/public/catalogue_v2')
    p.add_argument('--data-dir',default='data/public/expanded_v2')
    p.add_argument('--recipes',default='configs/public_expression_recipes_v2.json')
    p.add_argument('--run-dir')
    p.add_argument('--data-only',action='store_true')
    args=p.parse_args()
    path=build_data(args.data_dir,args.panel_dir,args.catalogue_dir,args.recipes)
    if args.data_only:
        print('CONVERTED',path,flush=True);return
    c=read_json(args.source_config)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run=Path(args.run_dir or 'runs/public_expanded_'+stamp).resolve()
    if run.exists():raise FileExistsError('Preserve old run; use new directory')
    run.mkdir(parents=True)
    c.update(manifest=str(path),run_dir=str(run),pilot_only=False,defer_public_test=False,
             independent_dataset_cards=True,internal_final_test_only=True,
             reuse_smoke_config=str(Path(args.source_config).resolve()))
    write_json(run/'config.json',c)
    print('PREPARE grouped candidates and SFT examples',flush=True)
    audit(c);prepare(c);supervision_audit(c);quality=export_text(c)
    if not quality['train']['positive_gene_targets']:raise ValueError('No supervised genes; training is not ready')
    write_json(run/'preparation_status.json',{'status':'prepared','label_quality':quality,
        'tests_executed':False,'new_smoke_executed':False,'qwen_training_executed':False,'internal_data_used':False})
    print('TRAIN_CONFIG='+str(run/'config.json'),flush=True)
    Path(args.data_dir,'latest_training_config.txt').write_text(str(run/'config.json')+'\n',encoding='utf-8')


if __name__=='__main__':main()
