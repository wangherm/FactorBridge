"""Download the complete declared GEO catalogue; keep acquisition separate from admission.

No FASTQ alignment, implicit matrix conversion, internal file discovery, or training.
Every series gets a status, including failures. Exit 2 means incomplete acquisition.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factorbridge.io import read_json, write_json, sha256
from factorbridge.public_data import download_verified


def acquire(url, path, expected=None, max_bytes=0):
    path = Path(path)
    receipt = path.with_name(path.name + '.receipt.json')
    if path.exists():
        actual = sha256(path)
        previous = read_json(receipt) if receipt.exists() else {}
        if not expected and not previous.get('sha256'):
            raise ValueError(f"Unverified pre-existing file: {path}")
        if actual != (expected or previous['sha256']):
            raise ValueError(f"Cache checksum mismatch: {path}")
        return {'path': str(path), 'url': url, 'sha256': actual, 'bytes': path.stat().st_size, 'status': 'verified_cache'}
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'FactorBridge-public-acquisition/0.2'})
            with urllib.request.urlopen(req, timeout=90) as r:
                size = int(r.headers.get('Content-Length') or 0)
                if max_bytes and size > max_bytes:
                    raise ValueError(f"Explicit size limit: {size} > {max_bytes}")
                if size and size + 2 * 1024**3 > shutil.disk_usage(path.parent).free:
                    raise ValueError(f"Insufficient free disk for {size} bytes plus reserve")
                total = 0
                with part.open('wb') as out:
                    while chunk := r.read(1024 * 1024):
                        total += len(chunk)
                        if max_bytes and total > max_bytes:
                            raise ValueError('Explicit size limit exceeded')
                        out.write(chunk)
                if size and total != size:
                    raise IOError(f"Short download: {total}/{size}")
            actual = sha256(part)
            if expected and actual != expected:
                raise ValueError('Source checksum differs from pinned observed bytes')
            part.replace(path)
            result = {'path': str(path), 'url': url, 'sha256': actual, 'bytes': total, 'status': 'downloaded',
                      'checksum_basis': 'pinned_observed_source' if expected else 'first_download_observed_not_publisher_signature'}
            write_json(receipt, result)
            return result
        except ValueError:
            raise
        except Exception as exc:
            last = exc
            if attempt < 2: time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Download failed after 3 attempts: {url}: {last}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue', default='configs/public_catalogue_v2.json')
    p.add_argument('--destination', default='data/public/catalogue_v2')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-file-gib', type=float, default=0, help='0: no per-file cap; limited files are explicitly incomplete')
    p.add_argument('--metadata-only', action='store_true')
    args = p.parse_args()
    if args.workers < 1 or args.max_file_gib < 0: p.error('Invalid workers or size limit')
    root = Path(args.destination).resolve(); root.mkdir(parents=True, exist_ok=True)
    catalogue = read_json(args.catalogue)
    def series(spec):
        acc = spec['accession']; folder = root / acc
        out = {'accession': acc, 'files': [], 'errors': [], 'training_admission': 'requires_explicit_conversion_and_unit_audit'}
        base = spec['base_url']
        try:
            out['files'].append(acquire(base+'soft/'+acc+'_family.soft.gz', folder/(acc+'_family.soft.gz'), spec.get('metadata_sha256')))
            # Fetch a fresh index: unavailable/unpublished sources remain errors.
            index = folder/'index.html'
            info = acquire(base+'suppl/', index)
            out['files'].append(info)
            names = [n for n in re.findall(r'href="([^"]+)"', index.read_text()) if n.startswith(acc+'_') or n=='filelist.txt']
            pins = {r['filename']: r.get('sha256') for r in spec.get('files', [])}
            out['discovered_files'] = names
            if not args.metadata_only:
                for name in names:
                    try:
                        print(f'DOWNLOAD {acc}/{name}', flush=True)
                        r = acquire(base+'suppl/'+name, folder/name, pins.get(name), int(args.max_file_gib*1024**3))
                        out['files'].append(r)
                        print(f"{r['status'].upper()} {acc}/{name}: {r['bytes']} bytes", flush=True)
                    except Exception as exc:
                        out['errors'].append({'filename': name, 'error': str(exc)})
        except Exception as exc:
            out['errors'].append({'stage': 'metadata', 'error': str(exc)})
        out['status'] = 'incomplete' if out['errors'] else 'metadata_only' if args.metadata_only else 'series_supplements_downloaded'
        write_json(folder/'acquisition.json', out)
        return out
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(series, s) for s in catalogue['geo_series']]
        for f in as_completed(futures):
            result = f.result(); results.append(result)
            write_json(root/'geo_acquisition.json', {'series': results, 'complete': False, 'internal_data_used': False})
            print(f"SERIES {len(results)}/{len(futures)} {result['accession']}: {result['status']}", flush=True)
    unresolved = [r for r in catalogue['records'] if r['catalogue_record'] > 3 and not any(a.startswith('GSE') for a in r['accessions'])]
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'geo_series': results,
              'non_geo_records_require_repository_specific_acquisition': unresolved,
              'internal_killifish': 'not_accessed_frozen_final_test', 'training_executed': False,
              'scope': 'GEO series SOFT and deposited supplementary files, including mixed modalities; not SRA raw reads',
              'all_catalogue_data_downloaded': False,
              'geo_acquisition_complete': not args.metadata_only and not any(r['errors'] for r in results)}
    write_json(root/'acquisition_report.json', report)
    print(json.dumps({'report': str(root/'acquisition_report.json'), 'geo_acquisition_complete': report['geo_acquisition_complete'],
                      'non_geo_unresolved_records': len(unresolved)}, indent=2))
    if not report['geo_acquisition_complete'] or unresolved: sys.exit(2)


if __name__ == '__main__': main()
