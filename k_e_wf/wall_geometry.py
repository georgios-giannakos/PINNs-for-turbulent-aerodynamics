# SPDX-License-Identifier: Apache-2.0
"""Offset of the unchanged TMR Warp polygon, fixed nominal convex profile."""
import numpy as np
from custom_airfoil_geometry import AirfoilInChannel


class WallFunctionGeometry(AirfoilInChannel):
    def __init__(self,*args,wall_offset,**kwargs):
        super().__init__(*args,**kwargs)
        self.wall_offset=float(wall_offset)
        if not 0<self.wall_offset<0.01*self.chord:
            raise ValueError("Wall offset must be in (0,0.01 chord).")
        # Wider y bounds isolate true solid distance from top/bottom distances.
        self.distance_geometry=AirfoilInChannel(
            float(self.lim[0]),float(self.lim[1]),-10*self.chord,10*self.chord,
            params=self.params,chord=self.chord,loc_camber=self.loc,
            airfoil_resolution=self.n,device=self.device)
        self.vertices,self.measures=self.polygon()
        a=self.vertices.astype(np.float64)
        e=np.roll(a,-1,axis=0)-a
        self.lengths=np.linalg.norm(e,axis=1)
        self.outward=np.column_stack((e[:,1],-e[:,0]))/self.lengths[:,None]
        prev=np.roll(self.outward,1,axis=0)
        turns=np.arctan2(prev[:,0]*self.outward[:,1]-prev[:,1]*self.outward[:,0],
                         np.sum(prev*self.outward,axis=1))
        # Tolerate tiny single-precision vertex roundoff only.
        if turns.min()<-0.03 or abs(np.maximum(turns,0).sum()-2*np.pi)>0.15:
            raise ValueError("Offset requires a convex nominal polygon.")
        self.offset_cumulative=np.cumsum(np.r_[self.lengths,self.wall_offset*np.maximum(turns,0)])
        d=self.wall_offset
        if (a[:,0].min()-d<=float(self.lim[0]) or a[:,0].max()+d>=float(self.lim[1])
            or a[:,1].min()-d<=float(self.lim[2]) or a[:,1].max()+d>=float(self.lim[3])):
            raise ValueError("Offset solid must fit in near rectangle.")
        rect=(float(self.lim[1])-float(self.lim[0]))*(float(self.lim[3])-float(self.lim[2]))
        self.resolved_area=float(rect-self.measures[0]-d*self.measures[1]-np.pi*d*d)
        if self.resolved_area<=0:
            raise ValueError("Nonpositive resolved fluid area.")

    def _check_parameters(self,p):
        p=self.params if p is None else p
        if p is not None:
            for v in p.param_ranges.values():
                if isinstance(v,tuple) and v[0]!=v[1]:
                    raise ValueError("WF does not support parameter ranges.")
                if isinstance(v,np.ndarray) and np.ptp(v)!=0:
                    raise ValueError("WF does not support parameter ranges.")
        vals=self._parameters(2,p,np.random.default_rng(13))
        if not np.allclose(vals,[0,0,0.12],atol=1e-8,rtol=0):
            raise ValueError("Fixed nominal NACA 0012 only.")

    def _map(self,wall,outward,area):
        point=wall+self.wall_offset*outward
        actual=self.distance_geometry.fluid_sdf(point)[:,0]
        if np.any(actual<=0) or not np.allclose(actual,self.wall_offset,rtol=.01,atol=5e-7):
            raise RuntimeError("Invalid physical distance at P.")
        tangent=np.column_stack((outward[:,1],-outward[:,0]))
        tangent*=np.where(tangent[:,0:1]>=0,1.,-1.)
        n=len(point)
        out=dict(x=point[:,0],y=point[:,1],x_wall=wall[:,0],y_wall=wall[:,1],
                 normal_x=-outward[:,0],normal_y=-outward[:,1],
                 tangent_x=tangent[:,0],tangent_y=tangent[:,1],
                 normal_distance=actual,area=area,alpha=np.zeros(n),
                 camber=np.zeros(n),thickness=np.full(n,0.12))
        return {k:np.asarray(v,dtype=np.float32).reshape(n,1) for k,v in out.items()}

    def sample_boundary(self,n_points,rand_seed=0,criteria=None,parameterization=None,quasirandom=False):
        if criteria is not None or quasirandom:
            raise ValueError("Use unfiltered pseudorandom offset sampling.")
        self._check_parameters(parameterization)
        n=int(n_points)
        if n<1:
            raise ValueError("Positive count required.")
        s=np.random.default_rng(int(rand_seed)%2147483647).uniform(0,self.offset_cumulative[-1],n)
        ids=np.searchsorted(self.offset_cumulative,s,side="right")
        local=s-np.r_[0.,self.offset_cumulative[:-1]][ids]
        wall=np.empty((n,2)); normal=np.empty((n,2))
        ns=len(self.vertices); a=self.vertices.astype(np.float64)
        seg=ids<ns; j=ids[seg]
        wall[seg]=a[j]+(local[seg]/self.lengths[j])[:,None]*(a[(j+1)%ns]-a[j])
        normal[seg]=self.outward[j]
        # Round vertex arcs close the computational offset contour, including TE.
        j=ids[~seg]-ns
        wall[~seg]=a[j]
        phi=np.arctan2(self.outward[(j-1)%ns,1],self.outward[(j-1)%ns,0])+local[~seg]/self.wall_offset
        normal[~seg]=np.column_stack((np.cos(phi),np.sin(phi)))
        return self._map(wall,normal,np.full(n,self.offset_cumulative[-1]/n))

    def sample_physical_wall(self,n_points,rand_seed=0,parameterization=None):
        self._check_parameters(parameterization)
        raw=super().sample_boundary(n_points,rand_seed=rand_seed,parameterization=parameterization)
        wall=np.column_stack((raw["x"][:,0],raw["y"][:,0]))
        outward=-np.column_stack((raw["normal_x"][:,0],raw["normal_y"][:,0]))
        return self._map(wall,outward,raw["area"])

    def sample_interior(self,n_points,rand_seed=0,bounds=None,criteria=None,
                        parameterization=None,compute_sdf_derivatives=False,quasirandom=False):
        self._check_parameters(parameterization)
        n=int(n_points)
        if n<1:
            raise ValueError("Positive count required.")
        chunks=[]; remaining=n
        for i in range(20):
            raw=super().sample_interior(remaining+max(64,remaining//20),
                rand_seed=(int(rand_seed)+104729*i)%2147483647,bounds=bounds,criteria=criteria,
                parameterization=parameterization,compute_sdf_derivatives=compute_sdf_derivatives,
                quasirandom=quasirandom)
            q=np.column_stack((raw["x"][:,0],raw["y"][:,0]))
            sd=self.distance_geometry.fluid_sdf(q)
            idx=np.flatnonzero(sd[:,0]>self.wall_offset)[:remaining]
            if len(idx):
                data={key:val[idx].copy() for key,val in raw.items()}
                data.update(wall_distance=sd[idx,0:1],wall_normal_x=sd[idx,1:2],wall_normal_y=sd[idx,2:3])
                chunks.append(data); remaining-=len(idx)
            if not remaining:
                break
        if remaining:
            raise RuntimeError("Incomplete resolved-fluid rejection sampling.")
        out={key:np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
        out["area"]=np.full((n,1),self.resolved_area/n,dtype=np.float32)
        return out
