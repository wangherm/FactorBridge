"""Apply a fixed selector to a new public expression dataset, without constructing labels."""
from pathlib import Path
import numpy as np
from .data import manifest,load_dataset,local_split,preprocess
from .factors import candidates,card,refit,deduplicate
from .io import read_json,write_json,write_jsonl,digest
from .llm import infer,prepared
from .selectors import select


def recover(c,source,destination,method='qwen_finetuned'):
    if method not in {'qwen_finetuned','qwen_frozen','stability','non_llm','semantic_prior','loading_refit'}:
        raise ValueError('Unknown recovery method')
    raw=read_json(source)['datasets']
    if len(raw)!=1 or raw[0]['role'] not in {'public_train','public_validation'} or raw[0]['source_kind']!='public_real':
        raise ValueError('One public real dataset required; frozen test access is excluded from deployment exploration')
    e=manifest(source)[0]; out=Path(destination)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    x,genes,samples,meta=load_dataset(e);x=preprocess(x,e['data_scale'])
    disc,held=local_split(meta,e['biological_unit_col'],c['local_holdout_fraction'],c['seed'])
    units=[meta[i][e['biological_unit_col']] for i in disc]
    w,z,mean,recurrence,stability=candidates(x[disc],units,c,c['seed'])
    if c.get('semantic'):
        from .semantic import programme_candidates,enrich_card
        w,z,recurrence,stability,_,added=programme_candidates(x[disc],units,genes,e,c,w,z,recurrence,stability,c['seed'])
        write_json(out/'candidate_provenance.json',added)
    cards=[]
    for j in range(w.shape[1]):
        evidence,idx=card(w[:,j],recurrence[:,j],stability[j],genes,e,c)
        if c.get('semantic'):evidence,idx=enrich_card(evidence,idx,w[:,j],recurrence[:,j],genes,e,c)
        cards.append({'example_id':f'candidate_{j:03d}','card':evidence})
    if method.startswith('qwen'):
        predictions=infer(c,cards,method)
    else:
        learned=None
        if method=='non_llm':
            model=read_json(Path(c['run_dir'])/'non_llm.json');_,index=prepared(c)
            if model['config_hash']!=digest(c) or model['prepared_hash']!=digest(index):raise ValueError('Stale non-LLM model')
            learned=model['model']
        predictions=[{'example_id':r['example_id'],'valid':True,'output':select(r['card'],method,c,learned)[0]} for r in cards]
    write_jsonl(out/'candidate_cards.jsonl',cards);write_jsonl(out/'predictions.jsonl',predictions)
    lookup={str(g):i for i,g in enumerate(genes)}; ws=[];zs=[];accepted=[]
    for j,pred in enumerate(predictions):
        if not pred['valid'] or pred['output']['decision']!='retain':continue
        evidence=cards[j]['card'];slots={g['slot_id']:g['feature_id'] for g in evidence['genes']}
        support=[lookup[slots[s]] for s in pred['output']['supported_gene_slots']]
        fw,fz,_,_=refit(x[disc],x[held],support,w[:,j])
        ws.append(fw);zs.append(fz);accepted.append(j)
    keep=deduplicate(ws,zs,c['dedup_cosine'])
    W=np.column_stack([ws[i] for i in keep]) if keep else np.zeros((len(genes),0))
    Z=(x-mean)@W
    np.savez_compressed(out/'factors.npz',W=W,Z=Z,mean=mean,genes=genes,samples=samples,
                        discovery=disc,heldout=held,support_mask=abs(W)>1e-12)
    write_jsonl(out/'factor_cards.jsonl',[{'factor_id':f'factor_{column:03d}','column':column,
        'artifact':'factors.npz','evidence':cards[accepted[i]]['card'],'selection':predictions[accepted[i]]['output'],
        'scope':'expression programme hypothesis, not verified causal biology'} for column,i in enumerate(keep)])
    report={'status':'completed','method':method,'factors':W.shape[1],'samples':len(samples),'genes':len(genes),
            'invalid_predictions':sum(not r['valid'] for r in predictions),'reference_labels_created':False,
            'internal_data_used':False,'config_hash':digest(c)}
    write_json(out/'report.json',report)
    return report
