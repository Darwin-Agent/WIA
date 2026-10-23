#!/usr/bin/env python3
"""Check the actually selected Swift loss implementation before workers start."""
import argparse
import inspect
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-user-loss', action='store_true')
    args = parser.parse_args()
    from swift.loss_scale.base import LossScale
    from swift.dataset.preprocessor.core import RowPreprocessor
    from swift.template import ContextType
    path = Path(inspect.getfile(LossScale)).resolve()
    alias_row = {'messages': [
        {'role': 'user', 'content': 'initial', 'loss_mask': False},
        {'role': 'assistant', 'content': 'action', 'loss_mask': True},
        {'role': 'user', 'content': 'observation', 'loss_mask': True},
        {'role': 'assistant', 'content': 'masked', 'loss_mask': False},
    ]}
    RowPreprocessor._check_messages(alias_row)
    if [message.get('loss') for message in alias_row['messages']] != [False, True, True, False]:
        raise RuntimeError('Swift drops loss_mask fields; apply the release loss-mask alias patch before training')
    if args.train_user_loss:
        os.environ['SWIFT_TRAIN_USER_LOSS'] = '1'
        contexts = ['<u>initial</u>', 'a0', '<u>observation</u>', 'a1']
        types = [ContextType.OTHER, ContextType.RESPONSE, ContextType.OTHER, ContextType.RESPONSE]
        messages = [{'role': 'user', 'content': 'initial', 'loss': False},
                    {'role': 'assistant', 'content': 'a0', 'loss': True},
                    {'role': 'user', 'content': 'observation', 'loss': True},
                    {'role': 'assistant', 'content': 'a1', 'loss': False}]
        segments, scales = LossScale()(contexts, types, messages)
        learned = ''.join(s for s, w in zip(segments, scales) if w > 0)
        if learned != 'a0observation':
            raise RuntimeError(f'Swift does not honor the expected message loss mask: {path}; apply runtime/patches')
    print(f'Swift loss implementation verified: {path}')

if __name__ == '__main__':
    main()
