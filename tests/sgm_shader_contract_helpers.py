"""Instantiate the legacy register capacity for frozen shader comparisons."""
import re


def legacy_specialized_body(source, text):
    start = source.index('void aggregate_specialized_body<')
    signature = source[start:source.index('{', start)]
    assert re.search(r'\blet\s+Registers\s*:\s*uint\s*=\s*8\s*>', signature), \
        'legacy register capacity must remain eight'
    text = re.sub(r',\s*let\s+Registers\s*:\s*uint\s*=\s*8\s*(?=>)', '', text)
    text = re.sub(r'\bRegisters\s*-\s*1\b', '7u', text)
    return re.sub(r'\bRegisters\b', '8', text)
