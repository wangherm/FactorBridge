"""Download real NCBI/Reactome evidence, freeze bytes, export a species-specific bundle.

The initial HTTPS download records observed hashes, not publisher-certified hashes.
Subsequent use verifies the exact cached bytes. No case-folded or fuzzy gene mapping.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
from pathlib import Path
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factorbridge.io import read_json, write_json, sha256

SPECIES = {'Homo sapiens': '9606', 'Mus musculus': '10090', 'Rattus norvegicus': '10116'}


def acquire(url, path):
    receipt = path.with_suffix(path.suffix+'.receipt.json')
    if path.exists():
        r = read_json(receipt)
        if r['url'] != url or r['sha256'] != sha256(path):
            raise ValueError(f'Cached evidence changed: {path}')
        print('VERIFIED CACHE', path.name, flush=True)
        return r
    part = path.with_suffix(path.suffix+'.part')
    print('DOWNLOAD', url, flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'FactorBridge/0.2'}), timeout=120) as source, part.open('wb') as out:
        while block := source.read(1024*1024):
            out.write(block)
        expected = source.headers.get('Content-Length')
        if expected and out.tell() != int(expected):
            raise IOError(f'Incomplete download: {url}')
        modified = source.headers.get('Last-Modified')
    r = {'url':url, 'sha256':sha256(part), 'retrieved_utc':datetime.now(timezone.utc).isoformat(),
         'last_modified':modified, 'hash_authority':'observed_https_bytes'}
    part.replace(path); write_json(receipt,r)
    print('VERIFIED DOWNLOAD', path.name, flush=True)
    return r


def build(destination):
    root = Path(destination); raw = root/'raw'; raw.mkdir(parents=True, exist_ok=True)
    output = root/'bundle.json'
    if output.exists():
        lock = read_json(root/'bundle.lock.json')
        if sha256(output) != lock['sha256']:
            raise ValueError('Annotation bundle changed')
        return output
    urls = {s.replace(' ','_')+'.gene_info.gz':
            'https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/'+s.replace(' ','_')+'.gene_info.gz' for s in SPECIES}
    urls.update({name:'https://reactome.org/download/current/'+name for name in
                 ['NCBI2Reactome.txt','pathway2summation.txt']})
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(lambda item: acquire(item[1], raw/item[0]), urls.items()))
    bundle = {'schema':'factorbridge-functional-v1', 'sources':receipts, 'species':{}, 'terms':{},
              'limitations':['RefSeq transcripts without a verified gene mapping remain unmapped',
              'Reactome non-human associations may be inferred electronically; preserve evidence codes',
              'NCBI gene descriptions and pathway memberships are priors, not expression measurements']}
    for species, taxid in SPECIES.items():
        genes, candidates = {}, {}
        with gzip.open(raw/(species.replace(' ','_')+'.gene_info.gz'), 'rt', encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if row['#tax_id'] != taxid:
                    # Species files also contain named strains/subspecies. Do not
                    # merge those symbols into the nominated reference taxon.
                    continue
                gid = row['GeneID']; key='NCBIGene:'+gid
                genes[key] = {'id':key, 'description':row['description'], 'terms':[]}
                # Official symbols and explicit Ensembl IDs only. Ambiguous aliases discarded.
                aliases = [row['Symbol'], gid] + [x.split(':',1)[1] for x in row['dbXrefs'].split('|') if x.startswith('Ensembl:')]
                for alias in aliases:
                    if alias and alias != '-':
                        candidates.setdefault(alias,set()).add(key)
        bundle['species'][species] = {'genes':genes, 'aliases':{k:next(iter(v)) for k,v in candidates.items() if len(v)==1},
                                     'ambiguous_alias_count':sum(len(v)>1 for v in candidates.values())}
    descriptions = {}
    with (raw/'pathway2summation.txt').open(encoding='utf-8') as f:
        for row in csv.reader(f,delimiter='\t'):
            # Export uses stable pathway ID in column one, possibly followed by a name.
            if row and re.fullmatch(r'R-[A-Z]{3}-\d+(?:\.\d+)?',row[0]):
                descriptions[row[0].split('.')[0]] = re.sub('<[^>]+>',' ',row[-1]).strip()
    with (raw/'NCBI2Reactome.txt').open(encoding='utf-8') as f:
        for row in csv.reader(f,delimiter='\t'):
            if len(row) != 6:
                raise ValueError('Unexpected Reactome mapping format')
            gid, term, url, name, evidence, species = row
            if species not in SPECIES:
                continue
            key='NCBIGene:'+gid
            if key not in bundle['species'][species]['genes']:
                continue
            bundle['species'][species]['genes'][key]['terms'].append(term)
            if term not in bundle['terms']:
                bundle['terms'][term]={'name':name,'description':descriptions.get(term,''),'source':url,'evidence_codes':[]}
            if evidence not in bundle['terms'][term]['evidence_codes']:
                bundle['terms'][term]['evidence_codes'].append(evidence)
    for species in bundle['species'].values():
        for record in species['genes'].values():
            record['terms']=sorted(set(record['terms']))
    if not bundle['terms']:
        raise ValueError('No functional mappings parsed')
    write_json(output,bundle)
    write_json(root/'bundle.lock.json',{'sha256':sha256(output),'sources':receipts,'terms':len(bundle['terms']),
        'species':{s:{'genes':len(d['genes']),'annotated_genes':sum(bool(g['terms']) for g in d['genes'].values())}
                   for s,d in bundle['species'].items()}})
    return output


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',default='data/annotations/ncbi_reactome_v1')
    a=p.parse_args(); print('ANNOTATION_BUNDLE='+str(build(a.destination).resolve()),flush=True)
