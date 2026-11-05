"""Task-local ordered four-field adapter; HP sampler exclusively owns indices."""
from bisect import bisect_right
import numpy as np

class OrderedFourFieldDataset:
    def __init__(self, prefixes, sequence_length=4096, pad_token=3):
        from hyper_parallel.data.tools.io import IndexedDataset
        if not prefixes or sequence_length < 1:
            raise ValueError('nonempty prefixes and positive sequence length required')
        self.sequence_length=sequence_length; self.pad_token=pad_token
        self.datasets=[IndexedDataset(path_prefix=str(p),multimodal=False,mmap=True) for p in prefixes]
        self.offsets=[0]
        for ds in self.datasets:self.offsets.append(self.offsets[-1]+len(ds))
    def __len__(self): return self.offsets[-1]
    def __getitem__(self,index):
        if index < 0:index += len(self)
        if not 0<=index<len(self):raise IndexError(index)
        part=bisect_right(self.offsets,index)-1
        text=self.datasets[part][index-self.offsets[part]]
        target=self.sequence_length+1
        if len(text)<target:text=np.pad(text,(0,target-len(text)),constant_values=self.pad_token)
        else:text=text[:target]
        tokens=text[:-1].copy();labels=text[1:].copy()
        mask=(labels!=self.pad_token).astype(np.int32)
        tokens[tokens==self.pad_token]=0;labels[labels==self.pad_token]=0
        pos=np.arange(self.sequence_length,dtype=np.int32)
        return tuple(np.asarray(x,dtype=np.int32) for x in (tokens,labels,mask,pos))

def collate_ordered(samples):
    """Name and stack fields; no shift, masking, or cursor changes."""
    import torch
    if not samples: raise ValueError('empty batch')
    names=('input_ids','labels','loss_mask','position_ids')
    return {name:torch.from_numpy(np.stack([sample[i] for sample in samples])) for i,name in enumerate(names)}

def build_ordered_loader(prefixes, *, consumed_samples=0, global_batch_size=2):
    """Single-rank LBS1, ordered sampler owns all cursor state."""
    from torch.utils.data import DataLoader
    from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
    dataset=OrderedFourFieldDataset(prefixes)
    sampler=build_dataset_batch_sampler(total_samples=len(dataset),micro_batch_size=1,
        global_batch_size=global_batch_size,dp_rank=0,dp_world_size=1,
        consumed_samples=consumed_samples,drop_last=True,sampler_type='single')
    return DataLoader(dataset,batch_sampler=sampler,collate_fn=collate_ordered,num_workers=0),sampler
