# SPDX-License-Identifier: Apache-2.0
"""Independent double-precision reference and actual Warp runtime tests."""
import argparse
from types import SimpleNamespace
import numpy as np
from custom_airfoil_geometry import AirfoilInChannel
from dynamic_sampling import CachedGeometry, refresh_due


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    g=AirfoilInChannel(-0.32,1.28,-0.5,0.5,device=args.device)
    x=0.5*(1-np.cos(np.linspace(0,np.pi,501)))
    y=0.594689181*(0.298222773*np.sqrt(x)-0.127125232*x-0.357907906*x**2+0.291984971*x**3-0.105174606*x**4)
    y[[0,-1]]=0
    ref=np.concatenate([np.column_stack([x,y])[::-1],np.column_stack([x,-y])[1:-1]])
    v,m=g.polygon()
    np.testing.assert_allclose(v,ref,atol=4e-7,rtol=0)
    area=np.sum((y[1:]+y[:-1])*np.diff(x))
    length=2*np.sum(np.hypot(np.diff(x),np.diff(y)))
    np.testing.assert_allclose(m,[area,length],rtol=2e-5)
    # TMR describes the thickness as approximately 0.11894c. Its published
    # polynomial gives 0.1189712189c on this 501-point cosine grid.
    # Test the polynomial-derived reference, not the rounded prose value.
    reference_thickness = 0.11897121886378026
    np.testing.assert_allclose(2*np.max(y), reference_thickness, atol=1e-9, rtol=0)
    np.testing.assert_allclose(np.ptp(v[:,1]), reference_thickness, atol=8e-7, rtol=0)
    rng=np.random.default_rng(42)
    q=np.concatenate([rng.uniform([-0.3,-0.49],[1.27,0.49],(1000,2)),[[0.3,0],[-0.002,0],[1.002,0]]])
    a=ref
    b=np.roll(ref,-1,axis=0)
    e=b-a
    l2=np.sum(e*e,axis=1)
    expected=[]
    for z in q:
        delta=z-a
        t=np.clip(np.sum(delta*e,axis=1)/l2,0,1)
        d=np.min(np.linalg.norm(delta-t[:,None]*e,axis=1))
        cross=((a[:,1]>z[1])!=(b[:,1]>z[1])) & (z[0]<a[:,0]+(z[1]-a[:,1])*e[:,0]/np.where(e[:,1]!=0,e[:,1],1))
        if np.count_nonzero(cross)%2:
            d=-d
        expected.append(min(d,z[1]+0.5,0.5-z[1]))
    np.testing.assert_allclose(g.fluid_sdf(q)[:,0],expected,atol=1e-6,rtol=0)
    b=g.sample_boundary(512,rand_seed=10)
    n=g.sample_interior(7000,rand_seed=11)
    xy=lambda z:np.column_stack([z["x"][:,0],z["y"][:,0]])
    np.testing.assert_allclose(g.fluid_sdf(xy(b))[:,0],0,atol=5e-7)
    assert np.all(g.fluid_sdf(xy(n))[:,0]>0)
    normal=np.column_stack([b["normal_x"][:,0],b["normal_y"][:,0]])
    np.testing.assert_allclose(np.linalg.norm(normal,axis=1),1,atol=2e-5)
    assert np.mean(g.fluid_sdf(xy(b)+1e-5*normal)[:,0]<0)>0.98
    np.testing.assert_allclose(b["area"].sum(),length,rtol=2e-5)
    np.testing.assert_allclose(n["area"].sum(),1.6-area,rtol=2e-5)
    ranges=SimpleNamespace(param_ranges={"alpha":(-0.04,0.04),"camber":(0.0,0.02),"thickness":(0.10,0.14)})
    varying=g.sample_interior(64,rand_seed=21,parameterization=ranges)
    assert np.ptp(varying["alpha"])>0.05
    assert np.ptp(varying["thickness"])>0.02
    for i in range(64):
        values=[float(varying[k][i,0]) for k in ("alpha","camber","thickness")]
        _,metric=g.polygon(*values)
        np.testing.assert_allclose(varying["area"][i,0]*64,1.6-metric[0],rtol=2e-5)
    cache=CachedGeometry(g,1701,explicit_seed=True)
    initial=cache.sample_interior(128)
    assert cache.sample_interior(128) is initial
    cache.reset(1)
    updated=cache.sample_interior(128)
    assert not np.array_equal(initial["x"],updated["x"])
    cache.reset(1)
    np.testing.assert_array_equal(updated["x"],cache.sample_interior(128)["x"])
    assert not refresh_due(0,1000) and not refresh_due(999,1000) and refresh_due(1000,1000)
    print(f"PASS: geometry and cache on {args.device}; area={area:.9f}, perimeter={length:.9f}")


if __name__=="__main__":
    main()
