# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Compare standard FP32 tile join algorithms on actual operands."""
import json,hashlib
from pathlib import Path
import numpy as np
from compare_ce226 import load_arrays
from replay_ce_sumx_ragged import reduce_tree
p=Path('/alignment/phase12');s=Path('/state/jt-recipe-20261014')
a=json.loads((p/'evidence/ce226-comparison.json').read_text())
mf=load_arrays([json.loads(x) for x in (p/'runs/mf-ce226/CE226_OPERANDS.jsonl').read_text().splitlines()][-1]['records'],s/'mf-ce226/deep','name')
records=[]
for name,mapping in a['mapping'].items():
 parts=np.load(s/'ce226-reduction-replay-r2'/('call'+str(mapping['ce_call'])+'-native-parts.npy'))
 expected=mf[name,'sum'].reshape(-1)
 accum=parts[:,0].copy();comp=np.zeros_like(accum)
 for i in range(1,22):
  y=parts[:,i]-comp;t=accum+y;comp=(t-accum)-y;accum=t
 candidates={'sequential':reduce_tree(parts,'sequential'),'reverse':reduce_tree(parts[:,::-1],'sequential'),'adjacent':reduce_tree(parts,'adjacent'),'halves':reduce_tree(parts,'halves'),'ragged':reduce_tree(parts,'ragged-first'),'float64':reduce_tree(parts,'float64'),'kahan':accum}
 for key,values in candidates.items():
  records.append({'name':name,'candidate':key,'different':int(np.count_nonzero(values!=expected)),'max_abs':float(np.max(np.abs(values.astype(np.float64)-expected)))})
report={'complete':True,'records':records,'scope':'Diagnostic candidate joins of actual22 partial sums; not proof of native implementation or a training fix.'}
(p/'evidence/ce226-join-candidates.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
