"""Apply the existing independent reader to a separately pinned candidate source."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--lp-false', required=True, type=Path)
parser.add_argument('--lp-true', required=True, type=Path)
parser.add_argument('--source-manifest-sha256', required=True)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
assert re.fullmatch('[0-9a-f]{64}', args.source_manifest_sha256)
spec = importlib.util.spec_from_file_location('frozen_native_pair_reader', HERE / 'analyze_native_pair.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
original_reference = json.loads((HERE / 'aux-native-reference.json').read_text())
reference = copy.deepcopy(original_reference)
# Source identity is an independent run precondition, not part of the oracle equation.
reference['source_manifest_sha256'] = args.source_manifest_sha256
assets = load_file(str(HERE / 'aux-native-fixture.safetensors'))
assert reader.sha(HERE / 'aux-native-fixture.safetensors') == reference['assets_sha256']
target = assets['independent_fp64_global_gradient']
false_rows, false_grad, false_proof = reader.read_case(args.lp_false, False, reference, target)
true_rows, true_grad, true_proof = reader.read_case(args.lp_true, True, reference, target)
tolerance = reference['predeclared_reference_relative_l2_tolerance']
result = {
    'status': 'candidate_native_AUX_independent_pair_readback_complete',
    'candidate_source_manifest_sha256': args.source_manifest_sha256,
    'independent_reference_original_source_header': original_reference['source_manifest_sha256'],
    'independent_reference_file_sha256': reader.sha(HERE / 'aux-native-reference.json'),
    'fixture_sha256': reference['assets_sha256'], 'rows': [*false_rows, *true_rows],
    'all_eight_gradients_match_1x_target': all(row['relL2_to_1x'] < tolerance for row in [*false_rows, *true_rows]),
    'paired': [{'rank': rank, 'LP_true_vs_LP_false': reader.compare(true_grad[rank], false_grad[rank])} for rank in range(4)],
    'LPfalse_proof': false_proof, 'LPtrue_proof': true_proof,
    'scope': 'Independent AUX-only native FSDP/normalizer result, no MoE/CE/full training acceptance or reference coefficient change.'}
args.output.mkdir(exist_ok=False, parents=True)
(args.output / 'candidate-native-aux-pair.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
