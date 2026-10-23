"""Shared CoT option parsing for local, streaming and API evaluation."""
import argparse


def parse_think_mode(value: str) -> str:
    mode = str(value).strip().lower()
    mode = {'1': 'on', 'true': 'on', '0': 'off', 'false': 'off'}.get(mode, mode)
    if mode not in {'on', 'off', 'auto'}:
        raise argparse.ArgumentTypeError('think-mode must be on/off/auto (1/0 and true/false are aliases)')
    return mode
