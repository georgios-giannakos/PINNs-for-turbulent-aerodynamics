# SPDX-License-Identifier: Apache-2.0
"""Equilibrium Launder-Spalding wall treatment with signed full stress."""
import math
import torch
from torch import nn

C_MU=0.09
KAPPA=0.4187
E=9.793
Y_JOIN=11.224708076046031
WALL_INPUTS=["u","v","k","ep","u__x","u__y","v__x","v__y",
             "normal_x","normal_y","tangent_x","tangent_y","normal_distance"]
WALL_OUTPUTS=["velocity_wall_normal_wf","velocity_wall_parallel_wf",
              "wall_shear_stress_wf","ep_wf","wall_tangential_velocity",
              "wall_tau_model","wall_tau_rans","u_tau","u_tau_shear",
              "y_plus_wall","u_plus_wall","ep_wall_target","cf_wall","nu_t_wall"]


def u_plus(yp):
    return torch.where(yp<=Y_JOIN,yp,torch.log(E*torch.clamp(yp,min=Y_JOIN))/KAPPA)


def epsilon_wall(k,nu,d,yp):
    y=torch.clamp(yp,min=0.,max=1e4)
    gamma=.001*y**4/(1+y)
    return (2*nu*k/d**2*torch.exp(-gamma)
            +C_MU**.75*k**1.5/(KAPPA*d)*torch.exp(-1/(gamma+1e-20)))


class WallFunction(nn.Module):
    def __init__(self,nu,utau_ref,d_ref):
        super().__init__()
        if min(nu,utau_ref,d_ref)<=0:
            raise ValueError("Positive wall references required.")
        self.nu=float(nu)
        self.tau_ref=float(utau_ref)**2
        self.ep_ref=float(utau_ref)**3/(KAPPA*float(d_ref))

    def forward(self,a):
        u,v,k,ep=(a[key] for key in ("u","v","k","ep"))
        nx,ny,tx,ty,d=(a[key] for key in
                        ("normal_x","normal_y","tangent_x","tangent_y","normal_distance"))
        ut=u*tx+v*ty; un=u*nx+v*ny
        utau=C_MU**.25*torch.sqrt(k)
        yp=utau*d/self.nu; up=u_plus(yp)
        ratio=torch.where(yp<=Y_JOIN,torch.ones_like(yp),yp/torch.clamp(up,min=1e-20))
        tau=self.nu*ut/d*ratio
        nt=C_MU*k**2/(ep+1e-20)
        ux,uy,vx,vy=(a[key] for key in ("u__x","u__y","v__x","v__y"))
        # n points INTO the solid; minus sign gives traction on the solid.
        tau_rans=-(self.nu+nt)*(tx*(2*ux*nx+(uy+vx)*ny)
                                +ty*((uy+vx)*nx+2*vy*ny))
        ep_target=epsilon_wall(k,self.nu,d,yp)
        return {
            "velocity_wall_normal_wf":un,
            "velocity_wall_parallel_wf":torch.abs(ut)-utau*up,
            "wall_shear_stress_wf":(tau_rans-tau)/self.tau_ref,
            "ep_wf":(ep-ep_target)/self.ep_ref,
            "wall_tangential_velocity":ut,"wall_tau_model":tau,
            "wall_tau_rans":tau_rans,"u_tau":utau,
            "u_tau_shear":torch.sqrt(torch.abs(tau)),
            "y_plus_wall":yp,"u_plus_wall":up,"ep_wall_target":ep_target,
            "cf_wall":2*tau,"nu_t_wall":nt,
        }


class NearWallInit(nn.Module):
    """Heuristic warm start only: near-wall equilibrium scales, far freestream."""
    def __init__(self,nu,offset,utau_ref,k_inf,ep_inf,width):
        super().__init__()
        if min(nu,offset,utau_ref,k_inf,ep_inf,width)<=0:
            raise ValueError("Positive initialization scales required.")
        self.nu,self.offset,self.utau_ref=float(nu),float(offset),float(utau_ref)
        self.k_inf,self.ep_inf,self.width=float(k_inf),float(ep_inf),float(width)

    def forward(self,a):
        d=a["wall_distance"]; nx,ny=a["wall_normal_x"],a["wall_normal_y"]
        fade=torch.exp(-((d-self.offset)/self.width)**2)
        local_utau=self.utau_ref*torch.abs(ny)
        kw=torch.maximum(local_utau**2/math.sqrt(C_MU),torch.full_like(d,self.k_inf))
        yp=C_MU**.25*torch.sqrt(kw)*self.offset/self.nu
        ew=torch.maximum(epsilon_wall(kw,self.nu,torch.full_like(d,self.offset),yp),
                         torch.full_like(d,self.ep_inf))
        yp0=self.utau_ref*self.offset/self.nu
        factor=min(1.,self.utau_ref*(yp0 if yp0<=Y_JOIN else math.log(E*yp0)/KAPPA))
        u0=(1-fade)+fade*factor*(1-nx**2); v0=-fade*factor*nx*ny
        k0=self.k_inf+fade*(kw-self.k_inf)
        ep0=self.ep_inf+fade*(ew-self.ep_inf)*self.offset/d
        return {"u_near_init":a["u"]-u0,"v_near_init":a["v"]-v0,
                "p_near_init":a["p"],"k_near_init":(a["k"]-k0)/k0,
                "ep_near_init":(a["ep"]-ep0)/ep0}
