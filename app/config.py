"""Small, validated configuration shared by the pipeline and HTTP worker."""
import json
import os
from functools import lru_cache
from pathlib import Path

DEFAULTS = {
    'nfe': 32,
    'cpu_threads': 4,
    'compile_threads': 8,
    'max_frames': 4096,
    'probe_frames': [128, 256, 512, 1024, 2048, 4096],
    'groups': ['small', 'medium', 'large'],
    'concurrencies': [1, 2, 8, 16, 32],
    'minimum_requests': 4,
    'warmups': 1,
    'quality_enabled': True,
    'numerical_l2_max': 0.02,
    'numerical_cosine_min': 0.999,
    'cer_ceiling': 0.10,
    'serve_backend': 'fp32_eager',
    'allow_unqualified_aoti': False,
}


def validate(values):
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown settings: {sorted(unknown)}')
    result = {**DEFAULTS, **values}
    for key in ('nfe', 'cpu_threads', 'compile_threads', 'minimum_requests', 'warmups'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(result['max_frames']) is not int or not 512 <= result['max_frames'] <= 4096:
        raise ValueError('max_frames must be between 512 and 4096')
    if not result['probe_frames'] or any(type(v) is not int or not 128 <= v <= result['max_frames'] for v in result['probe_frames']):
        raise ValueError('probe_frames must fit within 128..max_frames')
    for key in ('groups', 'concurrencies', 'probe_frames'):
        if len(result[key]) != len(set(result[key])):
            raise ValueError(f'{key} must not contain duplicates')
    if not result['groups'] or not set(result['groups']) <= {'small', 'medium', 'large'}:
        raise ValueError('groups must contain small, medium or large')
    if not result['concurrencies'] or any(type(v) is not int or not 1 <= v <= 64 for v in result['concurrencies']):
        raise ValueError('concurrencies must be integers between 1 and 64')
    for key in ('quality_enabled', 'allow_unqualified_aoti'):
        if type(result[key]) is not bool:
            raise ValueError(f'{key} must be true or false')
    if result['serve_backend'] not in ('fp32_eager', 'bf16_eager', 'bf16_aoti'):
        raise ValueError('Unknown serve_backend')
    for key in ('numerical_l2_max', 'cer_ceiling', 'numerical_cosine_min'):
        if not isinstance(result[key], (int, float)) or not 0 < result[key] <= 1:
            raise ValueError(f'{key} must be in (0,1]')
    return result


@lru_cache
def settings():
    filename = os.environ.get('CONFIG_FILE')
    values = json.loads(Path(filename).read_text()) if filename else {}
    return validate(values)
