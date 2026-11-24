# SPDX-License-Identifier: Apache-2.0
"""Single-worker periodic sampling cache."""
import numpy as np
import zlib


class CachedGeometry:
    def __init__(self,source,seed,measure=None,explicit_seed=False):
        self.source=source
        self.seed=int(seed)
        self.measure=measure
        self.explicit_seed=explicit_seed
        self.reset(0)

    @property
    def dims(self):
        return self.source.dims

    def reset(self,epoch):
        self.epoch=int(epoch)
        self.cache={}

    def _sample(self,kind,n_points,**kwargs):
        key=(kind,int(n_points))
        if key not in self.cache:
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,int(n_points)]))
            seed=int(rng.integers(1,2147483647))
            f=getattr(self.source,kind)
            if self.explicit_seed:
                data=f(n_points,rand_seed=seed,**kwargs)
            else:
                state=np.random.get_state()
                try:
                    np.random.seed(seed)
                    data=f(n_points,**kwargs)
                finally:
                    np.random.set_state(state)
            if self.measure is not None:
                data["area"]=np.full((int(n_points),1),self.measure/int(n_points),dtype=np.float32)
            self.cache[key]=data
        return self.cache[key]

    def sample_boundary(self,n_points,**kwargs):
        return self._sample("sample_boundary",n_points,**kwargs)

    def sample_interior(self,n_points,**kwargs):
        return self._sample("sample_interior",n_points,**kwargs)


def constraint_seed(name,base):
    return (int(base)+zlib.crc32(name.encode("utf-8")))%2147483647


def refresh_due(step,interval):
    return int(step)>0 and int(step)%int(interval)==0
