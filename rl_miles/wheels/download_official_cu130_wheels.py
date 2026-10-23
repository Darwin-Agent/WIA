#!/usr/bin/env python3
import json, os, shutil, subprocess, sys, urllib.request
TAG = os.environ.get('MILES_WHEELS_TAG', 'cu130-torch213-x86_64')
REPO = os.environ.get('MILES_WHEELS_REPO', 'yueming-yuan/miles-wheels')
DEST = os.environ.get('MILES_WHEELS_DEST', 'wheels')
ONLY = [item for item in os.environ.get('MILES_WHEELS_ONLY', '').split(',') if item]
url = f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}'
with urllib.request.urlopen(url, timeout=60) as response:
    data = json.load(response)
os.makedirs(DEST, exist_ok=True)
for asset in data.get('assets', []):
    name = asset['name']
    if not name.endswith(('.whl', '.tar.gz')):
        continue
    if ONLY and not any(name.startswith(prefix) for prefix in ONLY):
        print(f'skip {name}')
        continue
    out = os.path.join(DEST, name)
    expected_size = asset.get('size')
    if os.path.exists(out) and os.path.getsize(out) == expected_size:
        print(f'ok {name}')
        continue
    part = out + '.part'
    if os.path.exists(out) and os.path.getsize(out) != expected_size:
        os.replace(out, part)
    print(f'download {name}')
    if shutil.which('aria2c'):
        subprocess.run([
            'aria2c', '-x', '16', '-s', '16', '-k', '1M', '--continue=true',
            '--summary-interval=20', '--connect-timeout=10', '--timeout=30',
            '--max-tries=20', '--retry-wait=3', '-d', DEST, '-o', os.path.basename(part),
            asset['browser_download_url'],
        ], check=True)
    else:
        subprocess.run([
            'curl', '-fL', '--retry', '20', '--retry-delay', '5', '--connect-timeout', '20',
            '--speed-time', '300', '--speed-limit', '1024', '-C', '-', '-o', part,
            asset['browser_download_url'],
        ], check=True)
    if os.path.getsize(part) != expected_size:
        raise RuntimeError(f'incomplete download for {name}: {os.path.getsize(part)} != {expected_size}')
    os.replace(part, out)
