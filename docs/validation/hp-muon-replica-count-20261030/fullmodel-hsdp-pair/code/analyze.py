#!/usr/bin/env python3
"""Read one or two completed fixed HSDP replica-count runs; CPU/stdlib only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

RANKS = range(4)
STEPS = range(1501, 1541)
TIMED = range(1511, 1541)
FIELDS = {'input_ids', 'labels', 'loss_mask', 'position_ids'}
MEMORY = ('allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes')
COUNTS = {'model_local_states': 58, 'model_local_parameters': 57,
          'optimizer_local_main_parameters': 57, 'optimizer_local_slots': 86}
COMMON_PINS = {
    'source_manifest_sha256': 'f829c3146ec8981b0bc91ba87cc9f6c1cbcd3a3e04a563e7ab72a94b0ce4d4a1',
    'baseline_manifest_sha256': 'b18fef99d3d55649c439d865718db90cfc1d0b1f6742bad1f7c87fd7d9a9a8cb',
    'overlay_sha256': 'a3e5d62341c61309d8b7fb1261f2b4bb15022448e3430ea514ec5731c86ee265',
    'guard_sha256': 'e5a43e9fb4cb7b60f12f3a276f134bf43fcb09ed791d0d48bba5d7172516b4d2',
    'worker_sha256': 'bff3fa360e1b5d5cb43be4a8a57ece195c970f26369565a1b86f6b8ec200d047',
    'perf_sha256': '2b6f9bbdf164ec4fa75e48bdf7d52783e6b9c84ac3a3ee1f44d0a0ef01e0bc96',
    'helper_sha256': '11146f6773b57a7d0d5c10af802c9a625b86e7ccac63efa2adda293240eb24b9',
    'source_contract_sha256': '1fd4ac373a20f3edd8620cb0ac0593cf8f0ca5876a0d7f7227089358ef5cb633',
    'samples_sha256': '09e779b69af598c1bc04fef798b33a5fb4c47b04391f5b10388510aa1ee6e7b1',
    'historical_library_sha256': '6c53721550fc94d0f25f3d688b23290506d4f08821cc591bbb9c3f79caee8354',
}
ENV_SHA = 'e14f2704e4d568d79ef9df15546f3ed82e0613455ccb512f8c9e8c8b14b948cc'
ARMS = {
    'default': {'run_id': 'perf-hsdp-replica-default-125-r2', 'tag': 'hsdp-replica-default', 'count': None,
                'config_sha256': '9c5ae59905917bf43dce0bf04ff0537d6471f5088cbe9a5afb2cf25481bcc8bc',
                'scripts_manifest_sha256': '7efcafd354c129e5554fef915ec03e2733d4ce991532b519809816c08d5a6d4e'},
    'explicit1': {'run_id': 'perf-hsdp-replica-one-125-r2', 'tag': 'hsdp-replica-one', 'count': 1,
                  'config_sha256': '5ed73aafd36a973c4fa32a0f4ea54ea3c64da9e441d34a763b12df0c18f6f5ef',
                  'scripts_manifest_sha256': '8fd71a3b5caac7a6a3545e4c3328fe6887c85146fa279441fa05db2074d3bd08'},
}


class Incomplete(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Incomplete(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def hexsha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def reject_constant(value):
    raise Incomplete('non-finite JSON constant: ' + value)


def pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def parse_json(raw, label):
    try:
        return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=pairs_no_duplicates)
    except (ValueError, UnicodeError) as exc:
        raise Incomplete(label + ': invalid JSON: ' + str(exc)) from exc


class Archive:
    def __init__(self, root, readback=None):
        self.root = Path(root).resolve(strict=True)
        require(self.root.is_dir(), 'run root is not a directory')
        self.sources = {}
        self.inventory = None
        self.readback_sha256 = None
        if readback:
            raw = Path(readback).read_bytes()
            doc = parse_json(raw, 'readback')
            entries = doc.get('files')
            require(isinstance(entries, list), 'readback files inventory absent')
            self.inventory = {}
            for item in entries:
                name = item.get('path')
                require(isinstance(name, str) and name and not name.startswith('/')
                        and all(x not in ('', '.', '..') for x in name.split('/'))
                        and name not in self.inventory, 'invalid/duplicate readback path')
                require(type(item.get('size')) is int and item['size'] >= 0
                        and hexsha(item.get('sha256')), 'invalid readback file metadata')
                self.inventory[name] = item
            self.readback_sha256 = sha(raw)

    def read(self, name):
        path = self.root / name
        require(not path.is_symlink() and path.is_file(), 'missing/unsafe input: ' + name)
        raw = path.read_bytes()
        digest = sha(raw)
        if self.inventory is not None:
            entry = self.inventory.get(name)
            require(entry is not None and entry['size'] == len(raw) and entry['sha256'] == digest,
                    'readback mismatch: ' + name)
        self.sources[name] = {'sha256': digest, 'size': len(raw)}
        return raw

    def document(self, name):
        value = parse_json(self.read(name), name)
        require(isinstance(value, dict), 'expected JSON object: ' + name)
        return value


def load_canonical(path):
    raw = Path(path).read_bytes()
    require(sha(raw) == COMMON_PINS['samples_sha256'], 'canonical samples hash differs')
    doc = parse_json(raw, 'canonical')
    require(doc.get('first') == 6000 and doc.get('last') == 9999, 'canonical bounds differ')
    samples = doc['samples']
    for index in range(6000, 6008):
        sample = samples[str(index)]
        hashes = sample['field_sha256']
        require(sample['sample_index'] == index and set(hashes) == FIELDS
                and all(hexsha(v) for v in hashes.values()), 'invalid canonical sample')
    return doc


def check_pins(pre, start, arm):
    spec = ARMS[arm]
    require(pre.get('schema') == 'fullmodel-cpu-preflight-v1' and pre.get('status') == 'passed'
            and pre.get('arm') == arm and pre.get('run_id') == spec['run_id'], 'preflight identity')
    pins = dict(COMMON_PINS)
    pins.update({k: spec[k] for k in ('config_sha256', 'scripts_manifest_sha256')})
    require(all(pre.get(k) == v for k, v in pins.items()), 'preflight fixed pins differ')
    require(start.get('schema') == 'fullmodel-execute-start-v1' and start.get('arm') == arm
            and start.get('run_id') == spec['run_id'] and start.get('cid') == pre.get('cid'),
            'execute start identity')
    require(start.get('preflight_sha256') is not None
            and all(start.get('pins', {}).get(k) == v for k, v in pins.items()), 'start pins differ')
    require(start['pins'].get('host_sha256') == pre.get('host_sha256')
            == '7cab3afe82416aa4d9c219afae79d7d8699577bba84eeb65ee959995ca4fa583',
            'host pin differs')
    require(pre.get('source_files_verified') == 2661, 'preflight source inventory differs')
    require(pre['inputs']['environment_fingerprint']['sha256'] == ENV_SHA, 'environment pin differs')
    m = pre['python_metadata']['metadata']
    require(m['python'].startswith('3.12.14 ') and m['versions'] ==
            {'torch': '2.12.0+cpu', 'torch_npu': '2.12.0.post3', 'numpy': '1.26.4'},
            'runtime versions differ')
    require(m['origins']['hyper_parallel'].startswith('/work/sources/hyper-parallel/'),
            'actual import origin differs')
    return pins


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def time_statistics(values):
    require(len(values) == 30 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values),
            'expected 30 positive finite times')
    avg = statistics.fmean(values)
    return {'mean_seconds': avg, 'std_population_seconds': statistics.pstdev(values),
            'p50_seconds': percentile(values, .5), 'p90_seconds': percentile(values, .9),
            'min_seconds': min(values), 'max_seconds': max(values),
            'tokens_per_second_16384_over_mean': 16384 / avg,
            'percentile_method': 'linear interpolation at (n-1)*p'}


def read_perf(archive, rank):
    raw = archive.read('rank-%d-perf.jsonl' % rank)
    rows = {}
    for line, content in enumerate(raw.splitlines(), 1):
        require(bool(content.strip()), 'blank perf line')
        row = parse_json(content, 'perf line %d' % line)
        step = row.get('step')
        require(type(step) is int and step in STEPS and step not in rows and row.get('rank') == rank,
                'invalid/duplicate perf step or rank')
        require(row.get('phase') == ('warmup' if step <= 1510 else 'measured'), 'perf phase differs')
        for key, expected in (('global_batch_size', 4), ('micro_batch_size', 1),
                              ('num_micro_batches', 1), ('sequence_length', 4096),
                              ('global_tokens_per_update', 16384)):
            require(type(row.get(key)) is int and row[key] == expected, 'perf field differs: ' + key)
        dt = row.get('duration_seconds')
        require(type(dt) in (float, int) and math.isfinite(dt) and dt > 0, 'invalid duration')
        start, end = row.get('start_perf_counter_ns'), row.get('end_perf_counter_ns')
        require(type(start) is int and type(end) is int and end > start
                and math.isclose(dt, (end - start) * 1e-9, abs_tol=1e-9, rel_tol=1e-12),
                'duration and local timestamps disagree')
        for field in ('loss', 'grad_norm'):
            value = row.get(field)
            require(type(value) in (float, int) and math.isfinite(value), 'nonfinite step result')
        for which in ('memory_before', 'memory_after'):
            mem = row.get(which)
            require(isinstance(mem, dict) and all(type(mem.get(k)) is int and mem[k] >= 0 for k in MEMORY),
                    'invalid allocator memory record')
        rows[step] = row
    require(set(rows) == set(STEPS), 'perf step coverage differs')
    return rows


def check_batches(report, rank, canonical):
    expected_reads = [6000 + rank + 4 * i for i in range(40)]
    require(report.get('reads') == expected_reads, '40 actual sample indices differ')
    first = report.get('verified_first_batches')
    require(isinstance(first, list) and len(first) == 2, 'first two batch proofs missing')
    for offset, item in enumerate(first):
        step = 1501 + offset
        index = 6000 + rank + 4 * offset
        require(item.get('step') == step and item.get('micro') == 0 and item.get('sample_index') == index
                and item.get('dp_rank') == rank and item.get('cp_rank') == 0 and item.get('tp_rank') == 0
                and item.get('raw_draw_count') == 1 and item.get('local_slice') == [0, 4096]
                and item.get('prepared_four_fields_checked') is True, 'first batch metadata differs')
        require(item.get('full_field_sha256') == canonical['samples'][str(index)]['field_sha256'],
                'first batch field hashes differ from canonical')
    return [{'step': x['step'], 'sample_index': x['sample_index'],
             'field_sha256': x['full_field_sha256']} for x in first]


def layout_signature(record, *, ignore_replica_coord=False):
    require(isinstance(record, dict) and hexsha(record.get('local_sha256')), 'invalid tensor hash')
    for key in ('shape', 'global_shape', 'dtype', 'local_offset', 'placements', 'mesh'):
        require(key in record, 'tensor metadata missing: ' + key)
    mesh = record['mesh']
    if mesh is not None:
        require(isinstance(mesh, dict) and isinstance(mesh.get('coordinate'), list)
                and len(mesh['coordinate']) == 2, 'unsupported tensor mesh')
        mesh = dict(mesh)
        if ignore_replica_coord:
            mesh['coordinate'] = ['replica-ignored', mesh['coordinate'][1]]
    return {'shape': record['shape'], 'global_shape': record['global_shape'], 'dtype': record['dtype'],
            'local_offset': record['local_offset'], 'placements': record['placements'], 'mesh': mesh}


def observation_rows(obs, rank, count):
    require(obs.get('schema') == 'muon-assignment-metadata-v1' and obs.get('rank') == rank,
            'Muon observation identity')
    require(obs.get('configured_key_present') is (count is not None)
            and obs.get('configured_replica_count') == count and obs.get('effective_replica_count') == count
            and obs.get('explicit_flag') is (count is not None), 'Muon configured/effective mismatch')
    result = {}
    for group in obs['groups']:
        require(not group['no_comm'], 'no-comm Muon layout is outside this fixed experiment')
        for ass in group['assignments']:
            require(ass['replicate_sizes'] == ([1] if count == 1 else [2]), 'replica size differs')
            for rec in ass['records']:
                name = rec['fqn']
                require(isinstance(name, str) and name and name not in result, 'duplicate Muon FQN')
                bound = rec['binding_and_layout']
                layout = bound['main']
                require(layout['mesh_shape'] == [2, 2] and layout['mesh_rank_list'] == [0, 1, 2, 3]
                        and layout['mesh_coordinate'] == [rank // 2, rank % 2], 'actual mesh layout differs')
                places = layout['placements']
                require(isinstance(places, list) and len(places) == 2
                        and places[0]['class'].rsplit('.', 1)[-1] == 'Replicate'
                        and places[1]['class'].rsplit('.', 1)[-1] == 'Shard'
                        and places[1]['dim'] == 0, 'actual Muon placements differ')
                require(layout['mesh_replica_ranks_derived'] == [rank % 2, rank % 2 + 2]
                        and layout['mesh_shard_ranks_derived'] == [2 * (rank // 2), 2 * (rank // 2) + 1],
                        'derived mesh ranks differ')
                actual_repl = [rank] if count == 1 else [rank % 2, rank % 2 + 2]
                require(len(rec['replica_pgs']) == 1 and set(rec['replica_pgs'][0]) == set(actual_repl)
                        and len(rec['shard_pgs']) == 1
                        and set(rec['shard_pgs'][0]) == {2 * (rank // 2), 2 * (rank // 2) + 1},
                        'actual assignment PG differs')
                current = [0] if count == 1 else [rank // 2]
                owner = rec['owner_coord']
                require(ass['replicate_coordinate'] == current and isinstance(owner, list) and len(owner) == 1
                        and type(owner[0]) is int and 0 <= owner[0] < (1 if count == 1 else 2)
                        and type(rec['current_owned']) is bool
                        and rec['current_owned'] == (owner == current), 'owner mapping differs')
                require(type(rec['is_shard']) is bool, 'NS shard classification differs')
                result[name] = rec
    require(len(result) == obs['muon_parameter_count'] and result, 'Muon record coverage differs')
    return result


def check_group_progress(initial, final):
    require(isinstance(initial, list) and isinstance(final, list) and len(initial) == len(final),
            'optimizer group count differs')
    for before, after in zip(initial, final):
        require(before.get('optimizer_class') == after.get('optimizer_class')
                and before.get('members') == after.get('members'), 'optimizer group membership differs')
        left, right = dict(before['options']), dict(after['options'])
        require(left.pop('step', None) == 1500 and right.pop('step', None) == 1540
                and left == right, 'optimizer group progress/options differ')
        require(math.isclose(float(right['lr']), 1e-5, abs_tol=1e-12, rel_tol=0),
                'optimizer group learning rate differs')


def validate_rank(archive, arm, rank, canonical, audit, terminal_rows):
    spec = ARMS[arm]
    files = {'report': 'rank-%d-report.json' % rank, 'target': 'target-live-rank-%d.json' % rank,
             'initial': 'muon-initial-rank%d.json' % rank, 'post40': 'muon-post40-rank%d.json' % rank}
    docs = {key: archive.document(name) for key, name in files.items()}
    for key, name in files.items():
        require(terminal_rows['%s_sha256' % key] == archive.sources[name]['sha256'], 'terminal hash mismatch: ' + name)
    report, target, initial, post = (docs[k] for k in ('report', 'target', 'initial', 'post40'))
    require(report.get('status') == 'complete' and report.get('run_id') == spec['run_id']
            and report.get('rank') == rank and report.get('source_step') == 1500
            and report.get('global_step') == 1540 and report.get('training_updates') == 40
            and report.get('warmup_updates') == 10 and report.get('measured_updates') == 30
            and report.get('topology') == 'fsdp4' and report.get('strategy') == spec['tag']
            and report.get('checkpoint_saved') is False, 'rank report incomplete')
    require(target.get('schema') == 'native1500-final-short-target-v1'
            and target.get('run_id') == spec['run_id'] and target.get('source_step') == 1500
            and target.get('identity', {}).get('rank') == rank and target.get('topology') == 'fsdp4'
            and target.get('source_contract_sha256') == COMMON_PINS['source_contract_sha256'],
            'target identity differs')
    require({k: len(v) for k, v in target['tensors'].items()} == COUNTS, 'target 258 role counts differ')
    require(audit['target_rank_sha256'][str(rank)] == archive.sources[files['target']]['sha256'],
            'audit/target hash differs')
    require(initial.get('target_sha256') == archive.sources[files['target']]['sha256'], 'initial target binding')
    for doc, schema, step in ((initial, 'muon-initial-observation-v1', 1500),
                              (post, 'muon-post40-raw-observation-v1', 1540)):
        require(doc.get('schema') == schema and doc.get('run_id') == spec['run_id']
                and doc.get('rank') == rank and doc.get('global_step') == step
                and doc.get('audit_report_sha256') == archive.sources['root-external-audit-report.json']['sha256']
                and report.get('audit_report_sha256') == doc['audit_report_sha256'], 'Muon stage binding differs')
    ini = observation_rows(initial['muon'], rank, spec['count'])
    final = observation_rows(post['muon'], rank, spec['count'])
    require(set(ini) == set(final), 'Muon FQN membership changed')
    for name in ini:
        for key in ('record_index', 'owner_coord', 'current_owned', 'is_shard', 'replica_pgs', 'shard_pgs',
                    'binding_and_layout'):
            require(ini[name][key] == final[name][key], 'Muon assignment changed: ' + name)
    slots = post['raw_tensors']['optimizer_local_slots']
    links = post['momentum_by_owner']
    require(set(links) == set(final), 'Muon momentum join coverage differs')
    for name, rec in final.items():
        item = links[name]
        raw_present = rec['raw_state']['momentum_buffer_present']
        slot = slots.get(name + '.momentum_buffer')
        require(item.get('current_owned') is rec['current_owned']
                and item.get('momentum_present') is raw_present
                and item.get('momentum_record') == slot and (slot is not None) is raw_present,
                'Muon raw slot/hash join differs: ' + name)
        if slot is not None:
            layout_signature(slot)
    raw = post['raw_tensors']
    require(set(raw) == set(COUNTS) and all(
            set(raw[section]) == set(target['tensors'][section])
            for section in ('model_local_states', 'model_local_parameters', 'optimizer_local_main_parameters')),
            'post model/main coverage differs from initial 258 audit')
    groups = post['groups']
    require(groups == report['groups'], 'post40/report optimizer groups differ')
    check_group_progress(target['groups'], groups)
    muon_names, adam_names = set(), set()
    for group in groups:
        cls = group['optimizer_class']
        target_set = muon_names if cls.endswith('.muon.Muon') else adam_names if cls.endswith('.adamw.AdamW') else None
        require(target_set is not None, 'unsupported optimizer class')
        if target_set is adam_names:
            require(group['options'].get('amsgrad') is False, 'unsupported AdamW slot policy')
        for name in group['members']:
            require(name not in muon_names and name not in adam_names, 'duplicate group parameter')
            target_set.add(name)
    require(muon_names == set(final), 'Muon assignment/group FQNs differ')
    require(muon_names | adam_names == set(raw['model_local_parameters'])
            == set(raw['optimizer_local_main_parameters']), 'post groups/model membership differs')
    for name, rec in final.items():
        layout = rec['binding_and_layout']['main']
        captured = raw['optimizer_local_main_parameters'][name]
        layout_signature(captured)
        mesh = captured['mesh']
        require(mesh is not None and captured['shape'] == layout['local_shape']
                and captured['global_shape'] == layout['global_shape'] and captured['dtype'] == layout['dtype']
                and mesh['shape'] == layout['mesh_shape'] and mesh['rank_list'] == layout['mesh_rank_list']
                and mesh['coordinate'] == layout['mesh_coordinate'], 'Muon main/assignment layout differs')
        for observed, actual in zip(layout['placements'], mesh['placements']):
            require(all(observed.get(k) == actual.get(k) for k in ('class', 'repr', 'dim', 'uneven_shard')),
                    'Muon main/assignment placements differ')
        require(len(layout['placements']) == len(mesh['placements']), 'Muon main placement count differs')
    first = check_batches(report, rank, canonical)
    rows = read_perf(archive, rank)
    warm = archive.document('rank-%d-warm-storage.json' % rank)
    require(type(warm.get('unique_npu_storage_bytes')) is int and warm['unique_npu_storage_bytes'] >= 0,
            'warm storage is missing')
    return {'report': report, 'target': target, 'initial': initial, 'post': post, 'muon': final,
            'adam_names': adam_names, 'rows': rows, 'first_batches': first, 'warm': warm}


def read_arm(root, readback, arm, canonical):
    archive = Archive(root, readback)
    spec = ARMS[arm]
    pre = archive.document('cpu-preflight.json')
    start = archive.document('execute-start.json')
    pins = check_pins(pre, start, arm)
    require(start['preflight_sha256'] == archive.sources['cpu-preflight.json']['sha256'], 'preflight hash binding')
    term = archive.document('execute-terminal.json')
    require(term.get('status') == 'complete' and term.get('child_exit_code') == 0
            and term.get('run_id') == spec['run_id'] and term.get('cid') == pre.get('cid')
            and term.get('source_files_verified_after') == 2661, 'terminal incomplete')
    results = term.get('rank_outputs')
    require(isinstance(results, list) and len(results) == 4
            and set(x.get('rank') for x in results) == set(RANKS), 'terminal rank coverage')
    results = {x['rank']: x for x in results}
    audit = archive.document('root-external-audit-report.json')
    require(audit.get('status') == 'passed' and audit.get('run_id') == spec['run_id']
            and audit.get('source_step') == 1500 and audit.get('source_contract_sha256') == COMMON_PINS['source_contract_sha256']
            and audit.get('role_counts') == COUNTS and audit.get('source_rank_count') == 4
            and audit.get('target_rank_count') == 4 and len(audit.get('comparisons', [])) == 258
            and all(audit.get(x) is True for x in ('source_live_compared', 'target_live_compared', 'groups_compared',
                                                   'extra_compared', 'checkpoint_identity_stable')),
            'initial 258 audit incomplete')
    mapping = archive.document('preflight/resume5/physical-mapping-release.json')
    cards = mapping.get('physical_by_rank')
    require(mapping.get('run_id') == spec['run_id'] and cards == [4, 5, 6, 7], 'physical mapping differs')
    ranks = {r: validate_rank(archive, arm, r, canonical, audit, results[r]) for r in RANKS}
    proofs = mapping.get('rank_proof')
    require(isinstance(proofs, list) and len(proofs) == 4, 'physical mapping proof absent')
    by_rank = {x['rank']: x for x in proofs}
    require(set(by_rank) == set(RANKS), 'physical mapping rank proof differs')
    for r in RANKS:
        ident = ranks[r]['target']['identity']
        require(by_rank[r]['namespace_pid'] == ident['pid'] and by_rank[r]['starttick'] == ident['starttick'],
                'physical mapping process identity differs')
    for r in RANKS:
        target = ranks[r]['target']['tensors']
        actual = {(section, name) for section, vals in target.items() for name in vals}
        reported = {(x['section'], x['name']) for x in audit['comparisons']}
        require(len(reported) == 258 and reported == actual, 'initial audit comparison coverage differs')
    curve = []
    for step in STEPS:
        times = [ranks[r]['rows'][step]['duration_seconds'] for r in RANKS]
        curve.append({'step': step, 'phase': 'warmup' if step <= 1510 else 'measured',
                      'per_rank_seconds': times, 'max_rank_seconds': max(times)})
    rank_memory = []
    for r in RANKS:
        rows = ranks[r]['rows']
        rank_memory.append({'rank': r, 'physical_card': cards[r],
                            'timed_memory_after_max_bytes': {k: max(rows[s]['memory_after'][k] for s in TIMED)
                                                             for k in MEMORY},
                            'warm_storage_unique_npu_bytes': ranks[r]['warm']['unique_npu_storage_bytes'],
                            'initial_allocator_bytes': ranks[r]['report']['initial_memory'],
                            'final_allocator_bytes': ranks[r]['report']['final_memory']})
    return {'name': arm, 'run_id': spec['run_id'], 'pins': pins, 'host_sha256': pre['host_sha256'],
            'stats': time_statistics([x['max_rank_seconds'] for x in curve if x['phase'] == 'measured']), 'curve': curve,
            'rank_memory': rank_memory, 'ranks': ranks, 'sources': archive.sources,
            'readback_sha256': archive.readback_sha256,
            'archive_integrity': 'selected files checked against supplied readback' if readback else
                                 'local self-hashes only; independent readback not supplied'}


def compare_records(left, right):
    if left is None and right is None:
        return 'missing_both'
    if left is None:
        return 'missing_default'
    if right is None:
        return 'missing_explicit1'
    if layout_signature(left) != layout_signature(right):
        return 'layout_mismatch'
    return 'same_hash' if left['local_sha256'] == right['local_sha256'] else 'different_hash'


def compare_tensor_sections(default, explicit):
    results = {}
    for section in ('model_local_states', 'model_local_parameters', 'optimizer_local_main_parameters', 'adam_slots'):
        entries = []
        for rank in RANKS:
            a = default['ranks'][rank]['post']['raw_tensors']
            b = explicit['ranks'][rank]['post']['raw_tensors']
            if section == 'adam_slots':
                def select(run, rank, tensors):
                    names = run['ranks'][rank]['adam_names']
                    return {key: val for key, val in tensors['optimizer_local_slots'].items()
                            if key.rsplit('.', 1)[0] in names}
                left, right = select(default, rank, a), select(explicit, rank, b)
            else:
                left, right = a[section], b[section]
            keys = set(left) | set(right)
            if section == 'adam_slots':
                keys |= {name + '.' + slot for name in default['ranks'][rank]['adam_names']
                         for slot in ('exp_avg', 'exp_avg_sq')}
            for key in sorted(keys):
                entries.append({'rank': rank, 'name': key,
                                'result': compare_records(left.get(key), right.get(key))})
        summary = {k: sum(x['result'] == k for x in entries)
                   for k in ('same_hash', 'different_hash', 'missing_default', 'missing_explicit1',
                             'missing_both', 'layout_mismatch')}
        results[section] = {'counts': summary, 'records': entries}
    return results


def compare_muon_owners(default, explicit):
    records = []
    names = set(default['ranks'][0]['muon'])
    require(all(set(run['ranks'][r]['muon']) == names for run in (default, explicit) for r in RANKS),
            'Muon rank/arm FQN coverage differs')
    for shard in (0, 1):
        pair = (shard, shard + 2)
        for name in sorted(names):
            default_rows = [default['ranks'][r]['muon'][name] for r in pair]
            require(default_rows[0]['owner_coord'] == default_rows[1]['owner_coord'],
                    'replica peers disagree on owner')
            owners = [r for r in pair if default['ranks'][r]['muon'][name]['current_owned']]
            require(len(owners) == 1, 'expected one default owner per same-shard replica group')
            owner = owners[0]
            left = default['ranks'][owner]['post']['momentum_by_owner'][name]['momentum_record']
            for rank in pair:
                require(explicit['ranks'][rank]['muon'][name]['current_owned'],
                        'explicit1 rank does not own Muon record')
                right = explicit['ranks'][rank]['post']['momentum_by_owner'][name]['momentum_record']
                if left is None or right is None:
                    status = 'missing_both' if left is None and right is None else (
                        'missing_default_owner' if left is None else 'missing_explicit1_owner')
                elif (layout_signature(left) != layout_signature(
                        default['ranks'][owner]['post']['raw_tensors']['optimizer_local_main_parameters'][name])
                      or layout_signature(right) != layout_signature(
                        explicit['ranks'][rank]['post']['raw_tensors']['optimizer_local_main_parameters'][name])
                      or layout_signature(left, ignore_replica_coord=True) !=
                         layout_signature(right, ignore_replica_coord=True)):
                    status = 'layout_mismatch'
                else:
                    status = 'same_hash' if left['local_sha256'] == right['local_sha256'] else 'different_hash'
                records.append({'name': name, 'shard_coordinate': shard,
                                'default_owner_rank': owner, 'explicit1_owner_rank': rank, 'result': status})
    statuses = ('same_hash', 'different_hash', 'missing_default_owner', 'missing_explicit1_owner',
                'missing_both', 'layout_mismatch')
    return {'counts': {k: sum(x['result'] == k for x in records) for k in statuses}, 'records': records,
            'scope': 'Only default actual owner is read for each shard; default non-owner slot is ignored.'}


def analyze(default_root, explicit_root, canonical_path, default_readback=None, explicit_readback=None):
    canonical = load_canonical(canonical_path)
    a = read_arm(default_root, default_readback, 'default', canonical)
    b = read_arm(explicit_root, explicit_readback, 'explicit1', canonical)
    require(a['host_sha256'] == b['host_sha256'], 'host versions differ')
    for r in RANKS:
        require(a['ranks'][r]['first_batches'] == b['ranks'][r]['first_batches'], 'cross-arm samples differ')
        require(a['ranks'][r]['adam_names'] == b['ranks'][r]['adam_names'], 'Adam membership differs')
        require(a['ranks'][r]['target']['groups'] == b['ranks'][r]['target']['groups']
                and a['ranks'][r]['post']['groups'] == b['ranks'][r]['post']['groups'],
                'cross-arm optimizer groups differ')
    compare = {'tensor_hashes_same_rank_same_layout': compare_tensor_sections(a, b),
               'muon_momentum_actual_owners_same_shard': compare_muon_owners(a, b)}
    def compact(run):
        return {k: run[k] for k in ('run_id', 'pins', 'stats', 'curve', 'rank_memory',
                                    'sources', 'readback_sha256', 'archive_integrity')}
    default_mean = a['stats']['mean_seconds']
    explicit_mean = b['stats']['mean_seconds']
    compare['measured_time_ratio_explicit1_over_default'] = explicit_mean / default_mean
    compare['nominal_throughput_change_percent'] = 100 * (default_mean / explicit_mean - 1)
    return {'status': 'complete', 'default': compact(a), 'explicit1': compact(b), 'comparison': compare,
            'scope': {'timing': 'max of four local durations per step, not global makespan; 30 measured steps',
                      'tokens': '16384 nominal tokens per update, not loss-mask token count',
                      'hashes': 'same-layout local bytes only; hash differences are not numeric error or correctness',
                      'muon': 'default non-owner state ignored; absence never synthesized or equated'}}


def single_muon_owners(run, arm):
    """Summarize actual owners without inventing or comparing missing tensor bytes."""
    names = set(run['ranks'][0]['muon'])
    require(names and all(set(run['ranks'][r]['muon']) == names for r in RANKS),
            'Muon rank FQN coverage differs')
    records = []
    for shard in (0, 1):
        pair = (shard, shard + 2)
        for name in sorted(names):
            observations = [run['ranks'][r]['muon'][name] for r in pair]
            if arm == 'default':
                require(observations[0]['owner_coord'] == observations[1]['owner_coord'],
                        'replica peers disagree on owner')
                owners = [r for r in pair if run['ranks'][r]['muon'][name]['current_owned']]
                require(len(owners) == 1, 'expected one default owner per same-shard replica group')
            else:
                require(arm == 'explicit1' and all(x['current_owned'] for x in observations),
                        'explicit1 rank does not own Muon record')
                owners = list(pair)
            for rank in owners:
                post = run['ranks'][rank]['post']
                slot = post['momentum_by_owner'][name]['momentum_record']
                main = post['raw_tensors']['optimizer_local_main_parameters'][name]
                main_layout = layout_signature(main)
                if slot is None:
                    state = 'missing'
                elif layout_signature(slot) != main_layout:
                    state = 'layout_mismatch'
                else:
                    state = 'present_same_layout_as_main'
                records.append({'name': name, 'shard_coordinate': shard, 'owner_rank': rank,
                                'owner_coord': run['ranks'][rank]['muon'][name]['owner_coord'],
                                'momentum_state': state,
                                'momentum_sha256': slot['local_sha256'] if slot is not None else None,
                                'main_sha256': main['local_sha256']})
    states = ('present_same_layout_as_main', 'missing', 'layout_mismatch')
    return {'parameter_count': len(names),
            'owner_count_by_rank': {str(r): sum(x['owner_rank'] == r for x in records) for r in RANKS},
            'momentum_counts': {k: sum(x['momentum_state'] == k for x in records) for k in states},
            'records': records,
            'scope': 'Actual owners only; no comparison with another arm or inference from hash.'}


def analyze_single(root, readback, arm, canonical_path):
    require(readback is not None, 'single-arm analysis requires independent archive readback')
    require(arm in ARMS, 'unknown arm')
    canonical = load_canonical(canonical_path)
    run = read_arm(root, readback, arm, canonical)
    compact = {k: run[k] for k in ('run_id', 'pins', 'host_sha256', 'stats', 'curve', 'rank_memory',
                                  'sources', 'readback_sha256', 'archive_integrity')}
    return {'status': 'complete', 'arm': arm, 'run': compact, 'muon_owners': single_muon_owners(run, arm),
            'scope': {'timing': 'max of four local durations per step, not global makespan; 30 measured steps',
                      'tokens': '16384 nominal tokens per update, not loss-mask token count',
                      'hashes': 'local bytes only; no cross-arm comparison or numeric accuracy claim',
                      'muon': 'owner records only; missing slots stay missing',
                      'archive': 'readback checks only files read by this analysis'}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--default', type=Path, help='complete default RUN directory for pair mode')
    p.add_argument('--explicit1', type=Path, help='complete explicit1 RUN directory for pair mode')
    p.add_argument('--arm', choices=tuple(ARMS), help='single-arm mode')
    p.add_argument('--run', type=Path, help='complete RUN for single-arm mode')
    p.add_argument('--readback', type=Path, help='independent readback for single-arm mode')
    p.add_argument('--canonical', required=True, type=Path, help='pinned canonical-1501-2500.json')
    p.add_argument('--default-readback', type=Path)
    p.add_argument('--explicit1-readback', type=Path)
    p.add_argument('--out', type=Path, help='new output JSON; default stdout')
    args = p.parse_args()
    try:
        if args.arm is not None or args.run is not None or args.readback is not None:
            require(args.arm is not None and args.run is not None and args.readback is not None,
                    'single-arm mode requires arm, run and readback')
            require(all(x is None for x in (args.default, args.explicit1, args.default_readback,
                                             args.explicit1_readback)), 'single/pair arguments cannot be mixed')
            roots = [args.run.resolve()]
            data = analyze_single(args.run, args.readback, args.arm, args.canonical)
        else:
            require(args.default is not None and args.explicit1 is not None, 'pair mode requires both RUNs')
            require(args.default.resolve() != args.explicit1.resolve(), 'run roots must differ')
            roots = [args.default.resolve(), args.explicit1.resolve()]
            data = analyze(args.default, args.explicit1, args.canonical, args.default_readback, args.explicit1_readback)
        text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.out:
            destination = args.out.resolve()
            require(all(root not in destination.parents and destination != root for root in roots),
                    'output must not be inside an input archive')
            with args.out.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(text)
            print(json.dumps({'status': 'complete', 'output': str(destination)}, ensure_ascii=False))
        else:
            print(text, end='')
    except (Incomplete, OSError, ValueError, KeyError, TypeError) as exc:
        print('incomplete: ' + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
