#!/usr/bin/env python3
"""Validate local RL dependencies and reward configuration; no service requests or GPU jobs."""
import argparse
import importlib
import inspect
from pathlib import Path
import sys
from urllib.parse import urlsplit


def flag(value):
    value = str(value).lower()
    if value not in {'1', '0', 'true', 'false', 'yes', 'no', 'on', 'off'}:
        raise ValueError(f'Invalid boolean: {value}')
    return value in {'1', 'true', 'yes', 'on'}


def validate_urls(value, label):
    urls = [v.strip() for v in value.split(',') if v.strip()]
    if not urls:
        raise ValueError(f'{label} is required')
    for url in urls:
        parsed = urlsplit(url)
        if '<' in url or '>' in url or parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            raise ValueError(f'Set {label} to your deployed service URL(s), not a placeholder')
        _ = parsed.port  # Reject invalid ports before Ray starts.


def validate_prompt_files(root):
    files = [root / 'judge_user_prompt.txt'] + [root / name / 'judge_system_prompt.txt'
             for name in ('android', 'mcp', 'os', 'search', 'swe', 'terminal', 'web')]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing judge templates: ' + ', '.join(missing))


def import_hook(path):
    module, name = path.rsplit('.', 1)
    hook = getattr(importlib.import_module(module), name)
    if not callable(hook):
        raise TypeError(f'Non-callable RL hook: {path}')
    return hook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('miles-root', 'megatron-root', 'sglang-root', 'prompt-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--driver', default='train_async.py')
    parser.add_argument('--data-source', required=True)
    parser.add_argument('--reward', default='world_model_rl.reward.custom_rm')
    parser.add_argument('--rollout', required=True)
    parser.add_argument('--eval-rollout', default='world_model_rl.eval_rollout.raw_eval_rollout')
    parser.add_argument('--judge-enabled', default='1')
    parser.add_argument('--judge-base-url', default='')
    parser.add_argument('--judge-model', default='')
    parser.add_argument('--similarity-enabled', default='0')
    parser.add_argument('--similarity-url', default='')
    args = parser.parse_args()
    if flag(args.judge_enabled):
        validate_urls(args.judge_base_url, 'JUDGE_BASE_URL')
        if not args.judge_model or '<' in args.judge_model:
            raise ValueError('Set JUDGE_MODEL to the served judge model name')
        validate_prompt_files(args.prompt_root)
    if flag(args.similarity_enabled):
        validate_urls(args.similarity_url, 'SIMILARITY_URL')
    for path in (args.miles_root / args.driver, args.megatron_root / 'megatron/core/__init__.py',
                 args.sglang_root / 'python/sglang/__init__.py'):
        if not path.is_file():
            raise FileNotFoundError(f'Missing runtime source: {path}; run bootstrap_miles.sh and install_official_miles_env.sh')
    # Ensure imports resolve to the configured checkout, not a different pip installation.
    for module, root in [('miles', args.miles_root), ('megatron.core', args.megatron_root),
                         ('sglang', args.sglang_root)]:
        imported = importlib.import_module(module)
        if not Path(imported.__file__).resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f'{module} is imported from the wrong checkout')
    arguments = importlib.import_module('miles.utils.arguments')
    if '--think-mode' not in inspect.getsource(arguments):
        raise RuntimeError('Miles thinking CLI patch is missing; run bootstrap_miles.sh')
    for module in ('ray', 'torch', 'httpx', 'openai', 'transformers', 'transformer_engine.pytorch',
                   'fla', 'miles.backends.megatron_utils.model_provider', 'sglang.srt.models.qwen3_5'):
        importlib.import_module(module)
    for hook in (args.data_source, args.reward, args.rollout, args.eval_rollout):
        if hook:
            import_hook(hook)
    print('RL source imports, custom hooks, judge templates and service URL configuration verified.')

if __name__ == '__main__':
    main()
