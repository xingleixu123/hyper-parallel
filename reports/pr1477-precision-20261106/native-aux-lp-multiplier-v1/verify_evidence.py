"""Portable stdlib verification of actual synthetic AUX gradients; no Torch/NPU/API."""
import argparse,array,hashlib,json,math,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor(path,key):
    raw=path.read_bytes();n=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+n]);meta=header[key]
    assert meta['dtype'] in ('F32','F64');lo,hi=meta['data_offsets'];values=array.array('f' if meta['dtype']=='F32' else 'd')
    values.frombytes(raw[8+n+lo:8+n+hi])
    if sys.byteorder!='little':values.byteswap()
    assert len(values)==math.prod(meta['shape']) and all(map(math.isfinite,values))
    return meta,values
def stats(actual,expected):
    norm=math.fsum(v*v for v in expected);assert norm>0
    return dict(least_squares_multiplier=math.fsum(x*y for x,y in zip(actual,expected))/norm,
                relL2_vs_1x=math.sqrt(math.fsum((x-y)**2 for x,y in zip(actual,expected))/norm),
                relL2_vs_2x=math.sqrt(math.fsum((x-2*y)**2 for x,y in zip(actual,expected))/(4*norm)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);a=ap.parse_args()
    if (ROOT/'manifest.json').exists():
        manifest=json.loads((ROOT/'manifest.json').read_bytes())
        for name,entry in manifest['files'].items():
            relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
            path=ROOT/relative;assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],name
    p=ROOT/'model-audit/router-aux-native-reduction-probe'
    assert sha(p/'native-pair-a2/native-aux-pair.json')=='17bf7c48d1c7ed98d702a844f32ab61867037e3117c9f5c7632663b1b7f39c5b'
    assert sha(p/'root-native-pair-verified.json')=='435552c3235dbe763361d0c42303c2b91110fcd5fff01b0a0ed6e88e8da4f4e1'
    assert sha(p/'candidate-native-pair-a1/candidate-native-aux-pair.json')=='414316a040906619520a761c9af8ec79b43b81447ccb0329eb6204365833f7bc'
    assert sha(p/'aux-native-reference.json')=='ccda55fcad2ca45a64b5f07af98dfc5e8f232ff4ff25f2af945d25c52e14056f'
    reference=json.loads((p/'aux-native-reference.json').read_bytes())
    oracle=ROOT/'oracle/original-fp64-global-gradient.safetensors'
    assert sha(oracle)=='f13b52c2423c0e1880b110f4e64409924680bdcfac2361b177468222ef7e1b8f'
    meta,target=tensor(oracle,'independent_fp64_global_gradient');assert meta['dtype']=='F64' and meta['shape']==[64,3584]
    assert reference['not_MF_training_samples'] is True and reference['step_foundation_denominator']==3655
    case_index=json.loads((ROOT/'provenance/case-index.json').read_bytes());cases=case_index['cases']
    assert {(c['candidate'],c['loss_parallel']) for c in cases}=={(c,lp) for c in (False,True) for lp in (False,True)}
    rows=[];arrays={};digests={}
    for case in cases:
        candidate,lp=case['candidate'],case['loss_parallel'];folder=ROOT/case['relative_root']
        source='3292a5b93c7b62d4f97c3edf42572aeaca91a475057bb3540ca6bcf1b9eae245' if candidate else '4967881062f63d8144bd8c4214b60c16f9ca7495b7f241a6701d04a630e08d34'
        closure=json.loads((folder/'controller-closure.json').read_bytes())
        assert closure['derived_projection'] is True and closure['original_controller_retained_local_only'] is True
        assert closure['status']=='finished_process_exit_0' and closure['returncode']==0 and closure['holder_removed'] is True
        assert closure['source_before']==closure['source_after']=={'files':1544,'all_bytes_match':True}
        for rank in range(4):
            r=folder/('rank-'+str(rank));row=json.loads((r/'result.json').read_bytes());complete=json.loads((r/'complete.json').read_bytes())
            mapping=json.loads((r/'gradient-blob.json').read_bytes());blob=ROOT/mapping['included_blob']
            assert row['rank']==complete['rank']==rank and row['loss_parallel'] is lp and complete['loss_parallel'] is lp
            assert complete['result_sha256']==sha(r/'result.json') and row['run_id']==complete['run_id']==case['run_id']
            assert row['source_manifest_sha256']==case['source_manifest_sha256']==source and row['updates']==0
            assert row['manual_gradient_collectives_or_compensation'] is False
            assert row['fixture_sha256']=='a035d1bbb9afe561dc4a62f93586ae20653aa9a2f12d05d8760d7d4758e35d54'
            assert sha(blob)==mapping['sha256']==row['gradient_file_sha256'] and blob.stat().st_size==mapping['bytes']
            bm,before=tensor(blob,'native_before_step_token_normalize');am,actual=tensor(blob,'native_after_step_token_normalize')
            assert bm['dtype']==am['dtype']=='F32' and bm['shape']==am['shape']==row['native_chunk']['sizes']
            assert row['native_source_placements']==row['native_storage_source_placements']==['Replicate()']
            assert row['native_source_mesh']==row['native_storage_source_mesh']==['tp']
            assert row['actual_aux_groups']==[row['actual_CP_group']]
            assert len(row['actual_CP_group'])==len(row['actual_TP_group'])==2
            assert set(row['actual_CP_group'])&set(row['actual_TP_group'])=={rank}
            assert row['native_gradient_scaling_factor']==(.5 if lp else .25)
            for micro,record in enumerate(row['micros']):
                count=reference['foundation_counts_CP'][micro][row['actual_CP_group'].index(rank)]
                assert record['local_foundation_count']==count
                assert record['bound_foundation_multiplier']==record['actual_aux_context_scale']==2*count
                assert record['native_CP_loss_sum_multiplier']==2
            chunk=row['native_chunk'];assert chunk['offsets'][1]==0 and chunk['sizes'][1]==3584
            offset=chunk['offsets'][0]*3584;expected=target[offset:offset+len(actual)]
            assert len(expected)==len(actual) and max(abs(b/3655-a) for b,a in zip(before,actual))<1e-12
            found=stats(actual,expected);want='relL2_vs_1x' if candidate or not lp else 'relL2_vs_2x'
            assert found[want]<1e-5
            rows.append(dict(candidate=candidate,loss_parallel=lp,rank=rank,**found,blob_sha256=sha(blob)))
            arrays[candidate,lp,rank]=actual;digests[candidate,lp,rank]=sha(blob)
    for rank in range(4):
        assert all(t==2*f for f,t in zip(arrays[False,False,rank],arrays[False,True,rank]))
        assert digests[False,False,rank]==digests[True,False,rank]==digests[True,True,rank]
    result=dict(status='public_synthetic_AUX_raw_gradient_verification_passed',rows=rows,
                original_LP_true_equals_twice_false_elementwise_exact=True,
                candidate_both_LP_settings_byte_identical_to_original_false=True,
                actual_rank_results_verified=16,unique_native_gradient_blobs=4,
                whole_fixture_file_required_for_this_verifier=False,torch_or_HP_imported=False,NPU_used=False,
                scope='Actual synthetic router AUX-only native gradients; excludes complete MoE, main CE, optimizer, SP and training precision/performance acceptance.')
    if a.output:
        with a.output.open('x',encoding='utf8') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
