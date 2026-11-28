"""Launder-Sharma low-Re PINN in PyTorch; ep_tilde is NOT physical epsilon.

Equations: Launder & Sharma (1974); tensor E extension as in OpenFOAM.
No wall functions. Static p retains +2/3 grad(k) in momentum.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

def freestream(nu,intensity=.001,viscosity_ratio=10.):
    if nu<=0 or intensity<=0 or viscosity_ratio<=0:raise ValueError('Positive inlet values required')
    k=1.5*intensity**2
    # Solve .09*f_mu(Re_t)*Re_t = target nu_t/nu; using standard high-Re ep would miss the target.
    lo,hi=0.,max(100.,viscosity_ratio/.09)
    f=lambda r: .09*math.exp(-3.4/(1+r/50)**2)*r
    while f(hi)<viscosity_ratio:hi*=2
    for _ in range(80):
        mid=(lo+hi)/2
        if f(mid)<viscosity_ratio:lo=mid
        else:hi=mid
    rt=(lo+hi)/2
    return k,k*k/(nu*rt)

class MLP(nn.Module):
    def __init__(self,outputs,width,layers):
        super().__init__();mods=[nn.Linear(2,width),nn.Tanh()]
        for _ in range(layers-1):mods += [nn.Linear(width,width),nn.Tanh()]
        mods.append(nn.Linear(width,outputs));self.net=nn.Sequential(*mods)
        for m in self.modules():
            if isinstance(m,nn.Linear):nn.init.xavier_normal_(m.weight);nn.init.zeros_(m.bias)
        # Freestream initial state plus a small trainable spatial perturbation.
        nn.init.normal_(mods[-1].weight,std=1e-3)
    def forward(self,xy):return self.net(xy)

class FlowModel(nn.Module):
    def __init__(self,k_inf,et_inf,width=128,layers=5,k_scale=1e-3,et_scale=1e-2):
        super().__init__();self.flow=MLP(2,width,layers);self.pressure=MLP(1,width,layers)
        self.tke=MLP(1,width,layers);self.dissipation=MLP(1,width,layers)
        self.k_inf=k_inf;self.et_inf=et_inf;self.k_scale=k_scale;self.et_scale=et_scale
        self.k_bias=math.log(math.expm1(math.sqrt(k_inf/k_scale)))
        self.et_bias=math.log(math.expm1(math.sqrt(et_inf/et_scale)))
    def forward(self,xy):
        uv=self.flow(xy)
        q=math.sqrt(self.k_scale)*(F.softplus(self.tke(xy)+self.k_bias)+1e-15)
        e=math.sqrt(self.et_scale)*(F.softplus(self.dissipation(xy)+self.et_bias)+1e-15)
        return dict(u=1+uv[:,0:1],v=uv[:,1:2],p=self.pressure(xy),q=q,k=q*q,ep_tilde=e*e,ep_amp=e)

def grad(value,xy):
    return torch.autograd.grad(value,xy,torch.ones_like(value),create_graph=True,retain_graph=True)[0]

def physics(model,xy,nu,guard=1e-20):
    f=model(xy);u,v,p,q,k,et=[f[n] for n in ['u','v','p','q','k','ep_tilde']]
    gu,gv,gp,gq,gk,ge=[grad(f[n],xy) for n in ['u','v','p','q','k','ep_tilde']]
    ux,uy=gu[:,0:1],gu[:,1:2];vx,vy=gv[:,0:1],gv[:,1:2]
    rt=k*k/(nu*(et+guard));fmu=torch.exp(-3.4/(1+rt/50)**2);f2=1-.3*torch.exp(-rt**2)
    nt=.09*fmu*k*k/(et+guard);eff=nu+nt
    pk=nt*(2*ux*ux+2*vy*vy+(uy+vx)**2)
    D=2*nu*(gq*gq).sum(1,keepdim=True)
    gux,guy,gvx,gvy=[grad(z,xy) for z in [ux,uy,vx,vy]]
    # Sum over velocity components and ALL Hessian entries (mixed terms twice).
    E=2*nu*nt*sum((h*h).sum(1,keepdim=True) for h in [gux,guy,gvx,gvy])
    tx=grad(2*eff*ux,xy)[:,0:1]+grad(eff*(uy+vx),xy)[:,1:2]
    ty=grad(eff*(uy+vx),xy)[:,0:1]+grad(2*eff*vy,xy)[:,1:2]
    continuity=ux+vy
    mx=u*ux+v*uy+gp[:,0:1]+2*gk[:,0:1]/3-tx
    my=u*vx+v*vy+gp[:,1:2]+2*gk[:,1:2]/3-ty
    diffk=grad((nu+nt)*gk[:,0:1],xy)[:,0:1]+grad((nu+nt)*gk[:,1:2],xy)[:,1:2]
    diffe=grad((nu+nt/1.3)*ge[:,0:1],xy)[:,0:1]+grad((nu+nt/1.3)*ge[:,1:2],xy)[:,1:2]
    ak=u*gk[:,0:1]+v*gk[:,1:2];ae=u*ge[:,0:1]+v*ge[:,1:2]
    pe=1.44*pk*et/(k+guard);de=1.92*f2*et*et/(k+guard)
    rk=ak-diffk-pk+et+D;re=ae-diffe-pe+de-E
    # Detached local term scales improve conditioning; raw residuals are always exported.
    # The positive inlet scales prevent an unconstrained small-field denominator.
    sk=(abs(ak)+abs(diffk)+pk+et+D).detach()+model.k_inf
    se=(abs(ae)+abs(diffe)+pe+de+E).detach()+model.et_inf
    residual=dict(continuity=continuity,momentum_x=mx,momentum_y=my,k_equation=rk/sk,ep_equation=re/se)
    diagnostic=dict(f,nu_t=nt,nu_t_over_nu=nt/nu,Re_t=rt,f_mu=fmu,f_2=f2,production=pk,D=D,E=E,
                    ep=et+D,k_residual_raw=rk,ep_residual_raw=re,k_scale_local=sk,ep_scale_local=se,
                    u__x=ux,u__y=uy,v__x=vx,v__y=vy,**residual)
    return residual,diagnostic

def boundary_losses(model,wall,inlet,top_bottom,outlet):
    losses={}
    for name,xy in [('inlet',inlet),('outer',top_bottom)]:
        f=model(xy)
        losses[name+'_velocity']=((f['u']-1)**2+f['v']**2).mean()
        # Relative/log conditions retain O(1) cost for a 100% freestream error.
        losses[name+'_k']=torch.log(f['k']/model.k_inf).square().mean()
        losses[name+'_ep']=torch.log(f['ep_tilde']/model.et_inf).square().mean()
    f=model(wall)
    losses['wall_velocity']=(f['u']**2+f['v']**2).mean()
    losses['wall_q']=(f['q']/math.sqrt(model.k_inf)).square().mean()
    losses['wall_ep_amp']=(f['ep_amp']/math.sqrt(model.et_inf)).square().mean()
    f=model(outlet)
    losses['outlet_pressure']=f['p'].square().mean()
    # Diffusive transport needs an outlet condition; use zero normal gradients.
    losses['outlet_k_gradient']=(grad(f['k'],outlet)[:,0:1]/model.k_inf).square().mean()
    losses['outlet_ep_gradient']=(grad(f['ep_tilde'],outlet)[:,0:1]/model.et_inf).square().mean()
    return losses
