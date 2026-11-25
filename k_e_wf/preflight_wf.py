# SPDX-License-Identifier: Apache-2.0
"""Run in target Warp/PyTorch environment before PhysicsNeMo smoke test."""
import argparse
import numpy as np
import torch
from wall_geometry import WallFunctionGeometry
from wall_model import WallFunction,WALL_INPUTS,WALL_OUTPUTS,C_MU,u_plus,epsilon_wall,Y_JOIN
from dynamic_sampling import CachedGeometry


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",default="cpu")
    p.add_argument("--re",type=float,default=6e6)
    args=p.parse_args()
    nu=1/args.re; ur=.04; d=80*nu/ur
    g=WallFunctionGeometry(-.32,1.28,-.5,.5,wall_offset=d,device=args.device)
    b=g.sample_boundary(4096,rand_seed=18)
    w=g.sample_physical_wall(512,rand_seed=18)
    i=g.sample_interior(7000,rand_seed=20)
    assert np.all(i["wall_distance"]>d)
    np.testing.assert_allclose(i["area"].sum(),g.resolved_area,rtol=2e-6)
    np.testing.assert_allclose(b["area"].sum(),g.offset_cumulative[-1],rtol=2e-6)
    np.testing.assert_allclose(w["area"].sum(),g.measures[1],rtol=2e-5)
    for a in (b,w):
        np.testing.assert_allclose(a["normal_distance"],d,rtol=.01,atol=5e-7)
        np.testing.assert_allclose(a["normal_x"]*a["tangent_x"]+a["normal_y"]*a["tangent_y"],0,atol=2e-6)
        np.testing.assert_allclose(a["x"]-a["x_wall"],-d*a["normal_x"],atol=2e-7)
        np.testing.assert_allclose(a["y"]-a["y_wall"],-d*a["normal_y"],atol=2e-7)
    cache=CachedGeometry(g,1701,explicit_seed=True)
    before=cache.sample_boundary(512)
    assert cache.sample_boundary(512) is before
    cache.reset(1); after=cache.sample_boundary(512)
    assert not np.array_equal(before["x"],after["x"])
    cache.reset(1)
    np.testing.assert_array_equal(after["x"],cache.sample_boundary(512)["x"])
    m=WallFunction(nu,ur,d)
    T=lambda v:torch.tensor([[v]],dtype=torch.float64,device=args.device)
    for yp in (.01,1.,Y_JOIN,30.,80.,300.):
        for sign in (-1.,1.):
            f=yp*nu/d; k=f*f/C_MU**.5
            ep=epsilon_wall(T(k),nu,T(d),T(yp)).item()
            u=sign*f*u_plus(T(yp)).item()
            nt=C_MU*k*k/(ep+1e-20)
            a={key:T(0.) for key in WALL_INPUTS}
            a.update(u=T(u),k=T(k),ep=T(ep),normal_y=T(-1),tangent_x=T(1),
                     normal_distance=T(d),u__y=T(sign*f*f/(nu+nt)))
            out=m(a)
            assert set(out)==set(WALL_OUTPUTS)
            for key in WALL_OUTPUTS[:4]:
                assert abs(out[key].item())<1e-10,(yp,sign,key)
            assert np.sign(out["wall_tau_model"].item())==sign
            for key in ("u","k","ep","u__y"):
                a[key].requires_grad_(True)
            out=m(a)
            sum(out[key].square().mean() for key in WALL_OUTPUTS[:4]).backward()
            assert all(torch.isfinite(a[key].grad).all() for key in ("u","k","ep","u__y"))
    print(f"PASS: WF geometry/cache and manufactured signed residuals on {args.device}, d={d:g}")


if __name__=="__main__":
    main()
