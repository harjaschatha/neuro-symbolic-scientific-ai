#!/usr/bin/env python3
"""Render the documented research workflow as a code-native SVG."""
from html import escape
from pathlib import Path

steps = [
    ('Synthetic bioprocess trajectories', 'Multivariate inputs with known generating mechanisms'),
    ('Residual evidence and biological diagnostics', 'Temporal models and diagnostic rules provide supporting evidence'),
    ('Structured hypothesis', 'Classifier, rules, or LLM; source recorded per case'),
    ('Mechanistic fitting and guarded selection', 'Candidate ranking, constraints, and deterministic overrides'),
    ('Saved decisions and evaluation', 'Projection-aware, exact-label, and fixed-family metrics'),
]
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="770" viewBox="0 0 1000 770" role="img" aria-labelledby="title desc">',
         '<title id="title">Research workflow and portfolio scope</title>',
         '<desc id="desc">Five research stages followed by a note distinguishing the mock portfolio demo.</desc>',
         '<rect width="1000" height="770" fill="#101827"/>',
         '<g font-family="Arial, sans-serif" text-anchor="middle">',
         '<text x="500" y="48" fill="#ffffff" font-size="26">Research workflow: Final_Project</text>']
for i, (title, subtitle) in enumerate(steps):
    y = 78 + i * 115
    parts.extend([f'<rect x="75" y="{y}" width="850" height="86" rx="12" fill="#1e3048" stroke="#639dda"/>',
                  f'<text x="500" y="{y+33}" fill="#ffffff" font-size="21">{escape(title)}</text>',
                  f'<text x="500" y="{y+62}" fill="#c3d5e8" font-size="16">{escape(subtitle)}</text>'])
    if i < len(steps) - 1:
        parts.append(f'<path d="M500 {y+88} v19 m-6 -6 l6 6 6 -6" fill="none" stroke="#91b9df" stroke-width="2"/>')
parts.extend(['<text x="500" y="685" fill="#f6c66b" font-size="20">Portfolio demo: separate illustrative implementation</text>',
              '<text x="500" y="716" fill="#c3d5e8" font-size="17">Four growth curves · untrained GRU · mock reasoning</text>',
              '<text x="500" y="745" fill="#c3d5e8" font-size="16">Reported research metrics come from saved source artifacts.</text>', '</g></svg>'])
Path(__file__).with_name('system-overview.svg').write_text('\n'.join(parts) + '\n')
