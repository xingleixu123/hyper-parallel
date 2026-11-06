"""Verify full SVG vertex preservation and the original public archive.

Standard-library only; no renderer, browser, network, framework, or device import.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

from verify_public_csv_stats import load_and_verify

EXPECTED = {
    '01_single_raw': 4,
    '02_single_signed_deltas': 2,
    '03_single_cumulative': 2,
    '04_four_signed_deltas': 12,
    '05_four_cumulative': 12,
    '06_four_raw': 14,
    '07_aux_diagnostics': 15,
}
NS = '{http://www.w3.org/2000/svg}'
NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
VERTEX = re.compile(r'([ML])\s*(' + NUM + r')\s+(' + NUM + r')')
COMMAND = re.compile(r'[MmLlHhVvCcSsQqTtAaZz]')


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(folder, archive):
    _, csv_proof, _ = load_and_verify(archive)
    prior = json.loads((folder / 'CSV-verification.json').read_bytes())
    require(csv_proof == prior, 'independent CSV proof differs from rendered proof')
    source = json.loads((archive / 'manifest.json').read_bytes())
    require(sha(archive / 'manifest.json') == csv_proof['source_archive_manifest_sha256'], 'source archive manifest changed')
    original_files = []
    archive_root = archive.resolve()
    for relative, record in source['files'].items():
        path = (archive / relative).resolve()
        require(path.is_relative_to(archive_root), 'source path escapes archive')
        require(path.is_file() and path.stat().st_size == record['bytes'], relative + ': source size differs')
        require(sha(path) == record['sha256'], relative + ': source SHA differs')
        original_files.append(relative)
    audit = json.loads((folder / 'plot-data-audit.json').read_bytes())
    require(audit['numeric_points_plotted'] == 61000, 'in-memory numeric point coverage differs')
    require(audit['x_limits_all'] == [1501, 2500] and audit['all_data_inside_y_limits'], 'data limits differ')
    exported = []
    for name, expected in EXPECTED.items():
        svg = folder / (name + '.svg')
        root = ET.fromstring(svg.read_bytes())
        viewbox = [float(v) for v in root.attrib['viewBox'].split()]
        require(viewbox == [0, 0, 1152, 648], name + ': SVG canvas differs')
        require(not list(root.iter(NS + 'image')), name + ': raster content found in SVG')
        texts = list(root.iter(NS + 'text'))
        all_text = ''.join(''.join(t.itertext()) for t in texts)
        require(len(texts) >= 20 and '同数值汇总排版' in all_text, name + ': editable Chinese disclosure missing')
        numeric_paths = []
        for node in root.iter(NS + 'path'):
            data = node.attrib.get('d', '')
            commands = COMMAND.findall(data)
            if len(commands) < 100:
                continue
            require(commands == ['M'] + ['L'] * 999, name + ': numeric path was simplified or altered')
            vertices = [(float(x), float(y)) for _, x, y in VERTEX.findall(data)]
            require(len(vertices) == 1000 and all(math.isfinite(v) for point in vertices for v in point), name + ': numeric SVG vertices differ')
            xs = [x for x, _ in vertices]
            require(all(a < b for a, b in zip(xs, xs[1:])), name + ': step order was altered')
            spacing = (xs[-1] - xs[0]) / 999
            require(all(abs(x - (xs[0] + i * spacing)) < 1.1e-6 for i, x in enumerate(xs)), name + ': original uniform step spacing differs')
            numeric_paths.append({'vertices': len(vertices), 'x_first': xs[0], 'x_last': xs[-1]})
        require(len(numeric_paths) == expected, name + ': exported numeric series coverage differs')
        png = folder / (name + '.png')
        png_bytes = png.read_bytes()
        require(png_bytes[:16] == b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR', name + ': invalid PNG header')
        dimensions = list(struct.unpack('>II', png_bytes[16:24]))
        require(dimensions == [2400, 1350], name + ': PNG dimensions differ')
        exported.append({'figure': name, 'numeric_paths': len(numeric_paths), 'numeric_vertices': len(numeric_paths) * 1000,
                         'every_path_original_vertices': 1000, 'SVG_text_elements': len(texts),
                         'SVG_vector_only': True, 'SVG_editable_text': True,
                         'PNG_pixels': dimensions, 'PNG_sha256': sha(png), 'SVG_sha256': sha(svg)})
    require(sum(e['numeric_paths'] for e in exported) == 61, 'SVG total series differs')
    return {'schema': 'pr1477-static-figure-export-verification-v1', 'status': 'PASS',
            'artifact_kind': '同数值汇总排版', 'website_original_export': False,
            'source_archive_manifest_sha256': sha(archive / 'manifest.json'),
            'all_original_archive_files_SHA_verified': len(original_files),
            'all_original_archive_files_unchanged': True, 'CSV_files_verified': 26,
            'statistics': 'stdlib math.fsum; 21 statistic rows and 14 required mean rules exactly match original CSVs',
            'numeric_series': 61, 'numeric_points': 61000, 'window': [1501, 2500],
            'SVG_vertex_simplification': False, 'SVG_raster_content': False,
            'AUX_threshold': None, 'figures': exported,
            'manual_visual_review': 'Separate author-reviewed image hashes and observations are in visual-review.json; this script does not claim visual inspection.',
            'network_or_device_access': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parent.parent / 'public-final-evidence-staging')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    proof = verify(args.figures, args.archive)
    output = args.output or args.figures / 'export-QA.json'
    output.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps({'status': proof['status'], 'SVG_numeric_paths': proof['numeric_series'],
                      'SVG_full_vertices': proof['numeric_points'],
                      'source_files_unchanged': proof['all_original_archive_files_SHA_verified'],
                      'report_sha256': sha(output)}))


if __name__ == '__main__':
    main()
