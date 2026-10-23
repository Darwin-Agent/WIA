#!/usr/bin/env python3
"""Fetch pinned public sources and apply reviewed patches; --check is offline/read-only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT.parent

def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(['git', '-C', str(repo), *args], check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})

def apply_patches(repo: Path, patch_paths: list[Path], check_only: bool = False) -> None:
    # Check all patches before changing anything. Never reset or overwrite local work.
    pending = []
    for path in patch_paths:
        path = path.resolve()
        if git(repo, 'apply', '--reverse', '--check', str(path), check=False).returncode == 0:
            continue
        if check_only:
            raise RuntimeError(f'{path.name} is not applied in {repo}; run prepare_sources.py without --check')
        test = git(repo, 'apply', '--check', str(path), check=False)
        if test.returncode:
            raise RuntimeError(f'Cannot safely apply {path.name} in {repo}: {test.stderr.strip()}')
        pending.append(path)
    for path in pending:
        git(repo, 'apply', str(path))
        git(repo, 'apply', '--reverse', '--check', str(path))

def prepare(name: str, spec: dict, destination: Path, check_only: bool = False) -> None:
    if not (destination / '.git').exists():
        if check_only:
            raise RuntimeError(f'Missing {name} checkout: {destination}')
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f'Refusing to overwrite non-git directory: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', spec['url'], str(destination)], check=True)
        git(destination, 'checkout', '--detach', spec['revision'])
    actual = git(destination, 'rev-parse', 'HEAD').stdout.strip()
    if actual != spec['revision']:
        raise RuntimeError(f'{name}: expected {spec["revision"]}, got {actual}. Use a separate checkout; no automatic reset.')
    if not check_only:
        git(destination, 'submodule', 'update', '--init', '--recursive')
    apply_patches(destination, [ROOT / 'patches' / p for p in spec['patches']], check_only)
    print(f'{name}: pinned revision and {len(spec["patches"])} patch(es) verified')

def main() -> None:
    manifest = json.loads((ROOT / 'revisions.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=sorted(manifest), action='append', required=True)
    parser.add_argument('--destination', type=Path, help='override checkout directory (one component only)')
    parser.add_argument('--check', action='store_true', help='offline check; never clone, install or modify sources')
    args = parser.parse_args()
    if args.destination and len(args.component) != 1:
        parser.error('--destination requires exactly one --component')
    try:
        for name in args.component:
            spec = manifest[name]
            prepare(name, spec, args.destination or RELEASE / spec['path'], args.check)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'{exc}\n')

if __name__ == '__main__':
    main()
