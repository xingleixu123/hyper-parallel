# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Optimizer-independent two-process CPU diagnosis for issue 211."""

import argparse
from datetime import timedelta
import json

import torch
import torch.distributed as dist

from hyper_parallel import Shard, distribute_tensor, init_device_mesh


def main() -> None:
    """Compare reconstruction against a plain tensor with a bounded collective timeout."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, required=True)
    args = parser.parse_args()
    dist.init_process_group('gloo', timeout=timedelta(seconds=20))
    try:
        mesh = init_device_mesh('cpu', (2,), mesh_dim_names=('dp',))
        reference = torch.arange(args.rows * 64, dtype=torch.float32).reshape(args.rows, 64)
        value = distribute_tensor(reference, mesh, [Shard(0)])
        print(json.dumps({'stage': 'distributed', 'rank': dist.get_rank(),
                          'logical_shape': list(value.shape), 'local_shape': list(value.to_local().shape),
                          'placements': repr(value.placements)}), flush=True)
        for repetition in range(2):
            full = value.full_tensor()
            print(json.dumps({'stage': 'reconstructed', 'rank': dist.get_rank(),
                              'repetition': repetition, 'actual_shape': list(full.shape)}), flush=True)
            torch.testing.assert_close(full, reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
