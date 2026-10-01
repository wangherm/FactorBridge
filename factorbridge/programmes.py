"""Programme scores, semantic correspondence candidates and time holdout baselines.

No LLM-generated activity values, clocks or biological truths. Outer agents may
call the same API; the API exposes no private reference/label access.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import re
import numpy as np

from .io import read_json, read_jsonl, write_json, write_jsonl, sha256, digest
from .data import manifest, load_dataset
from .llm import prepared
from .semantic import bundle_for, resolve, normal_id


def rank_scores(x, gene_records, terms, min_members=3, min_coverage=0.1):
    """Tie-aware within-sample ranks, collapsed to unique mapped genes first.

    Returns transcript abundance enrichment scores, not protein/pathway activity.
    Missing programmes remain NaN accompanied by a coverage mask.
    """
    grouped = defaultdict(list)
    for i, r in enumerate(gene_records):
        grouped[r['id'] if r else 'unmapped:'+str(i)].append(i)
    groups = list(grouped.values())
    collapsed = np.column_stack([x[:, g].mean(axis=1) for g in groups])
    records = [gene_records[g[0]] for g in groups]
    membership=defaultdict(list)
    for i,r in enumerate(records):
        for term in r['terms'] if r else []:
            membership[term].append(i)
    ranks = np.empty_like(collapsed, dtype=float)
    for i, row in enumerate(collapsed):
        order = np.argsort(row, kind='stable')
        values = row[order]
        bounds = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1, len(row)]
        for start, end in zip(bounds[:-1], bounds[1:]):
            ranks[i,order[start:end]] = ((start+end-1)/2)/max(1,len(row)-1)-0.5
    scores = np.full((len(x),len(terms)),np.nan); coverage=[]
    for j, t in enumerate(terms):
        members = membership.get(t['term_id'],[])
        fraction = len(members)/max(1,t['annotated_members'])
        valid = len(members)>=min_members and fraction>=min_coverage
        coverage.append({'term_id':t['term_id'],'measured_members':len(members),'coverage':fraction,'available':valid})
        if valid:
            scores[:,j]=ranks[:,members].mean(axis=1)
    return scores, coverage


def _ridge(x,y,alpha=1.):
    mean=x.mean(0); scale=np.maximum(x.std(0),1e-8)
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    penalty=np.eye(design.shape[1])*alpha; penalty[0,0]=0
    beta=np.linalg.solve(design.T@design+penalty,design.T@y)
    return {'mean':mean,'scale':scale,'beta':beta}


def _predict(model,x):
    return np.column_stack([np.ones(len(x)),(x-model['mean'])/model['scale']])@model['beta']


def time_analysis(a,t,units,term_ids,seed=42,term_genes=None):
    """Future-time holdout with entire biological units kept together.

    Programme→coordinate head uses disjoint programmes from those used for
    deviation checking. No cross-species alignment is inferred from time scaling.
    """
    t=np.asarray(t,float); units=np.asarray(units)
    if len(t)!=len(a) or not np.isfinite(t).all():
        raise ValueError('Observed finite time coordinate required for every sample')
    times=np.unique(t)
    if len(times)<5:
        return {'status':'unavailable','reason':'fewer_than_five_observed_times'}
    boundary=times[max(3,int(len(times)*0.75))]
    future=t>=boundary
    future_units=set(units[future])
    fit=(~future)&np.array([u not in future_units for u in units])
    if len(np.unique(t[fit]))<3 or len(set(units[fit]))<4 or len(set(units[future]))<2:
        return {'status':'unavailable','reason':'insufficient_disjoint_units_before_and_after_time_boundary'}
    available=np.isfinite(a).all(0)
    valid=np.flatnonzero(available)
    valid=valid[np.std(a[fit][:,valid],axis=0)>1e-8]
    if len(valid)<4:
        return {'status':'unavailable','reason':'insufficient_covered_variable_programmes'}
    order=sorted(valid,key=lambda i:hashlib.sha256(term_ids[i].encode()).hexdigest())
    anchor=np.array(order[:min(4,max(1,len(order)//4))],dtype=int)
    used_genes=set().union(*(term_genes.get(term_ids[i],set()) for i in anchor)) if term_genes else set()
    checked=np.array([i for i in valid if i not in set(anchor) and
                      (not term_genes or not used_genes & term_genes.get(term_ids[i],set()))],dtype=int)
    if not len(anchor) or not len(checked):
        return {'status':'unavailable','reason':'empty_programme_partition'}
    origin=t[fit].min(); span=max(np.ptp(t[fit]),1e-8); u=(t-origin)/span
    clock=_ridge(a[fit][:,anchor],u[fit])
    estimated=_predict(clock,a[:,anchor])
    observed_design=np.column_stack([u,u*u])
    curve=_ridge(observed_design[fit],a[fit][:,checked])
    expected=_predict(curve,observed_design)
    residual=a[:,checked]-expected
    # Leave-one-biological-unit-out training residuals set predictive scale.
    loo=[]
    for unit in sorted(set(units[fit])):
        omitted=fit&(units==unit); remaining=fit&(units!=unit)
        if remaining.sum()>=4:
            m=_ridge(observed_design[remaining],a[remaining][:,checked])
            loo.append(a[omitted][:,checked]-_predict(m,observed_design[omitted]))
    if not loo:
        return {'status':'unavailable','reason':'no_grouped_uncertainty_estimate'}
    sigma=np.sqrt(np.mean(np.concatenate(loo)**2,axis=0))
    safe=sigma>1e-8
    deviations=np.full_like(residual,np.nan); deviations[:,safe]=residual[:,safe]/sigma[safe]
    mean=a[fit][:,checked].mean(0)
    last=a[fit & (t==t[fit].max())][:,checked].mean(0)
    trend=_ridge(u[fit,None],a[fit][:,checked]); trend_prediction=_predict(trend,u[:,None])
    # A small state-transition comparator: fit drift in a past-only latent space,
    # then roll forward without observing intermediate future expression.
    past_times=np.unique(u[fit])
    states=np.array([a[fit & (u==v)][:,checked].mean(0) for v in past_times])
    center=states.mean(0)
    _,singular,vt=np.linalg.svd(states-center,full_matrices=False)
    dimension=min(3,len(past_times)-2,int(np.sum(singular>1e-8)))
    transition_prediction=None
    if dimension:
        basis=vt[:dimension].T; latent=(states-center)@basis
        dt=np.diff(past_times)
        drift=_ridge(latent[:-1],np.diff(latent,axis=0)/dt[:,None])
        transition_prediction=np.tile(states[-1],(len(a),1))
        state=latent[-1:].copy();last_time=past_times[-1]
        for next_time in np.unique(u[future]):
            state=state+(next_time-last_time)*_predict(drift,state)
            transition_prediction[u==next_time]=(center+state@basis.T)[0]
            last_time=next_time
        if not np.isfinite(transition_prediction).all():raise ValueError('Unstable state-transition rollout')
    def mse(p): return float(np.mean((a[future][:,checked]-p[future])**2))
    result={'status':'executed','boundary':float(boundary),'fit_samples':int(fit.sum()),'future_samples':int(future.sum()),
        'future_coordinate_mae':float(np.mean(abs(estimated[future]-u[future]))),
        'forecast_mse':{'reference_mean':mse(np.tile(mean,(len(a),1))), 'last_observed_state':mse(np.tile(last,(len(a),1))),
                        'linear_time':mse(trend_prediction),'quadratic_reference':mse(expected)},
        'scope':'within_study_future_time_holdout; descriptive coordinate, not independent biological-clock validation',
        'clock_check_gene_overlap_removed':term_genes is not None,
        'limitations':['Rank normalization still shares the measured background; coordinate accuracy is not causal validation',
                      'Future coordinates extrapolate the fitted range; deviation z values are descriptive, not calibrated p values',
                      'Cross-sectional cohort ordering does not establish individual temporal dynamics'],
        'arrays':{'observed_coordinate':u,'estimated_coordinate':estimated,'expected':expected,'deviation':deviations,
                  'fit_mask':fit,'future_mask':future,'anchor_indices':anchor,'checked_indices':checked,
                  'reference_scale':sigma,'clock_beta':clock['beta'],'clock_mean':clock['mean'],'clock_scale':clock['scale'],
                  'curve_beta':curve['beta'],'curve_mean':curve['mean'],'curve_scale':curve['scale']}}
    if transition_prediction is not None:
        result['forecast_mse']['latent_state_transition']=mse(transition_prediction)
        result['arrays'].update(transition_prediction=transition_prediction,transition_basis=basis,
            transition_center=center,transition_beta=drift['beta'],transition_mean=drift['mean'],transition_scale=drift['scale'])
        result['transition_definition']='past cohort mean state -> ridge latent drift -> recursive future cohort state; no individual pairing'
    # Repeated residual covariance is fitted only on past samples, not on future deviations.
    from .factors import pca
    if np.linalg.norm(residual[fit])>1e-8:
        weights,_,center=pca(residual[fit],min(3,len(checked)),len(set(units[fit])))
        result['arrays'].update(residual_programme_loadings=weights,
                                residual_programme_scores=(residual-center)@weights,residual_center=center)
        result['residual_factor_status']='exploratory_past_fit_requires_independent_replication'
    return result


def observed_time(entry,meta):
    """Only explicit reviewed fields/units. No ordering inferred from row order."""
    study=entry['study_id']
    if study=='GSE124109' and all(r.get('starvation_days') is not None for r in meta):
        return np.array([float(r['starvation_days']) for r in meta]), 'serum_starvation_duration_days_not_depth'
    if study=='GSE132040':
        values=[re.search(r'(?:^|;)age=([0-9.]+) months postnatal(?:;|$)',r['condition']) for r in meta]
        if all(values): return np.array([float(v[1]) for v in values]), 'chronological_age_months'
    return None,'no_reviewed_time_recipe'


def run_programmes(c,split='validation'):
    if split not in {'train','validation'}:
        raise ValueError('Programme exploration cannot open frozen tests')
    if any(e['role'] not in {'public_train','public_validation'} or e['source_kind']=='internal_real'
           for e in read_json(c['manifest'])['datasets']):
        raise ValueError('Programme development requires a public train/validation-only manifest')
    root,index=prepared(c); bundle=bundle_for(c); run=Path(c['run_dir']); out=run/f'programmes_{split}'
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    entries={e['dataset_id']:e for e in manifest(c['manifest']) if e['role'] in {'public_train','public_validation'}}
    line=read_jsonl(root/'private/lineage.jsonl')
    # Fixed annotation vocabulary per species; rank by source size, never validation association.
    vocab={}
    for species,data in bundle['species'].items():
        counts=defaultdict(int)
        for record in data['genes'].values():
            for term in record['terms']: counts[term]+=1
        vocab[species]=[{'term_id':t,'annotated_members':n,'name':bundle['terms'][t]['name']}
                        for t,n in sorted(counts.items()) if 5<=n<=300]
    write_json(out/'vocabulary.json',vocab)
    reports=[]; profiles=[]
    for view_id in sorted({r['view_id'] for r in line if r['split']==split}):
        row=next(r for r in line if r['view_id']==view_id); e=entries[row['dataset_id']]
        with np.load(root/'private'/f'{view_id}.npz',allow_pickle=False) as f: view=dict(f)
        genes=view['genes'][view['measured']]; records=resolve(bundle,e['species'],genes)
        terms=vocab[e['species']]
        a,coverage=rank_scores(view['X'],records,terms)
        np.savez_compressed(out/f'{view_id}.npz',A=a,samples=view['samples'],term_ids=np.array([t['term_id'] for t in terms]),
                            available=np.array([r['available'] for r in coverage]), measured_genes=genes)
        write_json(out/f'{view_id}.coverage.json',coverage)
        report={'view_id':view_id,'dataset_id':e['dataset_id'],'study_id':e['study_id'],'noise_level':row['noise_level'],
                'available_programmes':sum(r['available'] for r in coverage),'total_programmes':len(terms),
                'score_definition':'mean centered within-sample gene rank; unsigned transcript abundance, not pathway activity'}
        # Metadata only: no reference matrices or training labels are opened here.
        from .data import table
        lookup={r['sample_id']:r for r in table(e['metadata_path'])}; meta=[lookup[s] for s in view['samples']]
        t,definition=observed_time(e,meta)
        if t is None:
            report['time_analysis']={'status':'unavailable','reason':definition}
        else:
            term_genes=defaultdict(set)
            for record in records:
                if record:
                    for term in record['terms']:term_genes[term].add(record['id'])
            result=time_analysis(a,t,[r[e['biological_unit_col']] for r in meta],[r['term_id'] for r in terms],c['seed'],term_genes)
            if 'arrays' in result:
                np.savez_compressed(out/f'{view_id}.time.npz',samples=view['samples'],**result.pop('arrays'))
            result['coordinate_definition']=definition; report['time_analysis']=result
        reports.append(report)
        print(f'PROGRAMMES {view_id}: {report["available_programmes"]}/{len(terms)}',flush=True)
    # Semantic correspondence is a candidate relation, never a gold identity label.
    evaluation=run/f'evaluation_{split}'/'factor_cards.jsonl'
    if evaluation.exists():
        factors=read_jsonl(evaluation)
        for factor in factors:
            card=factor['noisy_evidence']; funcs=card.get('functional_evidence',{})
            supported=set(factor['selected_support_genes'])
            slot_gene={g['slot_id']:g['feature_id'] for g in card['genes']}
            names={t['name']:sum(slot_gene[s] in supported for s in t['member_slots'])
                   for t in funcs.get('terms',[]) if t['q']<=c['semantic']['enrichment_fdr']}
            profiles.append({'factor_id':factor['factor_id'],'species':card['species'],'method':factor['method'],
                             'functions':{k:v for k,v in names.items() if v>=2}})
        pairs=[]
        for i,left in enumerate(profiles):
            for right in profiles[i+1:]:
                if left['species']==right['species'] or left['method']!=right['method']:continue
                a,b=set(left['functions']),set(right['functions']); shared=a&b
                if shared:pairs.append({'left':left['factor_id'],'right':right['factor_id'],'shared_functions':sorted(shared),
                    'functional_jaccard':len(shared)/len(a|b),'relation':'functional_correspondence_candidate_not_verified_identity'})
        write_jsonl(out/'cross_species_candidates.jsonl',pairs)
    write_json(out/'report.json',{'status':'completed','config_hash':digest(c),'annotation_sha256':c['semantic']['sha256'],
        'prepared_index_sha256':sha256(root/'index.json'),'datasets':reports,'internal_data_used':False,
        'world_model_executed':False,'agent_executed':False,'cross_species_factor_profiles':len(profiles)})
    return out


def agent_tool(operation,*,config,split='validation',source=None,destination=None,method='qwen_finetuned'):
    """Stable tool interface for a future outer agent; no autonomous loop or test access."""
    if split not in {'train','validation'}:
        raise ValueError('Frozen tests are not agent-accessible')
    if operation=='recover_expression':
        from .recovery import recover
        if not source or not destination:raise ValueError('Explicit source and destination required')
        return {'interface_version':1,'result':recover(config,source,destination,method)}
    if operation != 'programme_analysis':
        raise ValueError('Unknown version 1 operation')
    return {'interface_version':1,'artifact_directory':str(run_programmes(config,split))}
