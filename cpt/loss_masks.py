"""Normalize message-level binary loss_mask aliases to Swift's native loss field.

Masks are per message, NOT a token-level list. The initial user and system
messages must be masked out. Missing flags retain assistant-only semantics.
"""
from __future__ import annotations
from copy import deepcopy


def _binary(value, label):
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f'{label} must be a boolean or 0/1, not {type(value).__name__}')


def normalize_loss_masks(row: dict) -> dict:
    result = deepcopy(row)
    messages = result.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('Expected a nonempty messages list')
    row_mask = result.pop('loss_mask', None)
    if row_mask is not None and (not isinstance(row_mask, list) or len(row_mask) != len(messages)):
        raise ValueError('Row loss_mask must contain exactly one binary value per message (not per token)')
    seen_user = False
    for i, message in enumerate(messages):
        role = message.get('role')
        flags = []
        for field in ('loss', 'loss_mask'):
            if message.get(field) is not None:
                flags.append(_binary(message[field], f'messages[{i}].{field}'))
        if row_mask is not None:
            flags.append(_binary(row_mask[i], f'loss_mask[{i}]'))
        if flags and any(value != flags[0] for value in flags):
            raise ValueError(f'Conflicting loss and loss_mask values at message {i}')
        enabled = flags[0] if flags else role == 'assistant'
        initial_user = role == 'user' and not seen_user
        if role == 'user':
            seen_user = True
        if enabled and (role == 'system' or initial_user):
            raise ValueError(f'Message {i}: system and initial user must have loss=false')
        message.pop('loss_mask', None)
        message['loss'] = enabled
    return result
