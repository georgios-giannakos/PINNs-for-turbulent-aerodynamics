# SPDX-License-Identifier: Apache-2.0
from physicsnemo.sym.solver import Solver
from dynamic_sampling import CachedGeometry, constraint_seed, refresh_due
from custom_airfoil_geometry import AirfoilInChannel


class InferenceScheduledResamplingSolver(Solver):
    def __init__(self,cfg,domain,caches,interval):
        self.caches=caches
        self.interval=interval
        super().__init__(cfg,domain)

    def record_inferencers(self,step):
        result=super().record_inferencers(step)
        if refresh_due(step,self.interval):
            for cache in self.caches:
                cache.reset(int(step)//self.interval)
            print(f"[step {step}] Collocation caches refreshed; next batch uses a new seed.",flush=True)
        return result


def sampling_setup(cfg):
    interval=int(cfg.custom.resample_interval_steps)
    if interval<=0 or interval!=int(cfg.training.rec_inference_freq) or cfg.cuda_graphs:
        raise ValueError("Match inference and resample intervals; disable CUDA graphs.")
    caches=[]
    def cached(name,source,measure=None):
        c=CachedGeometry(source,constraint_seed(name,cfg.custom.sampling_seed),
                         measure,explicit_seed=isinstance(source,AirfoilInChannel))
        caches.append(c)
        return c
    return caches,interval,cached
