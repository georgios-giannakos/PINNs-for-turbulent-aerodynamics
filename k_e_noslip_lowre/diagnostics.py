"""Closed-surface force integration and portable ParaView XML exports."""
from pathlib import Path
import json
import numpy as np
import torch
from model import grad, physics

def write_vtp(path,xy,fields):
    n=len(xy);xyz=np.column_stack([xy,np.zeros(n)])
    def array(name,a,components=1,dtype='Float64'):
        a=np.asarray(a)
        return f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{components}" format="ascii">'+ ' '.join(map(str,a.reshape(-1)))+'</DataArray>'
    point=''.join(array(k,v) for k,v in fields.items())
    xml=f'<?xml version="1.0"?><VTKFile type="PolyData" version="0.1" byte_order="LittleEndian"><PolyData><Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0"><PointData>{point}</PointData><Points>'+array('Points',xyz,3)+'</Points><Verts>'+array('connectivity',np.arange(n),dtype='Int64')+array('offsets',np.arange(1,n+1),dtype='Int64')+'</Verts></Piece></PolyData></VTKFile>'
    Path(path).write_text(xml,encoding='utf8')

def wall_coefficients(fields,normal,ds,nu,p_inf=0.):
    nx,ny=normal.T;tx=np.abs(ny);ty=-nx*np.sign(ny)
    shear=nu*(fields['u__y']+fields['v__x'])
    fx=2*nu*fields['u__x']*nx+shear*ny
    fy=shear*nx+2*nu*fields['v__y']*ny
    cp=2*(fields['p']-p_inf);cf=2*(fx*tx+fy*ty)
    fp=-np.array([np.sum(cp*nx*ds),np.sum(cp*ny*ds)])
    fv=2*np.array([np.sum(fx*ds),np.sum(fy*ds)])
    return dict(Cp=cp,Cf=cf),dict(CL=float(fp[1]+fv[1]),CD=float(fp[0]+fv[0]),CL_p=float(fp[1]),CL_f=float(fv[1]),CD_p=float(fp[0]),CD_f=float(fv[0]))

def evaluate(model,xy,nu,device,batch=256,full=False):
    collected={}
    with torch.enable_grad():
        for start in range(0,len(xy),batch):
            q=torch.as_tensor(xy[start:start+batch],device=device,dtype=next(model.parameters()).dtype).requires_grad_(True)
            if full:_,f=physics(model,q,nu)
            else:
                f=model(q);gu=grad(f['u'],q);gv=grad(f['v'],q);gq=grad(f['q'],q)
                D=2*nu*gq.square().sum(1,keepdim=True)
                rt=f['k']**2/(nu*(f['ep_tilde']+1e-20));nt=.09*torch.exp(-3.4/(1+rt/50)**2)*f['k']**2/(f['ep_tilde']+1e-20)
                f.update(ep=f['ep_tilde']+D,D=D,nu_t_over_nu=nt/nu,u__x=gu[:,0:1],u__y=gu[:,1:2],v__x=gv[:,0:1],v__y=gv[:,1:2])
            for key,v in f.items():collected.setdefault(key,[]).append(v.detach().cpu().numpy().reshape(-1))
    return {k:np.concatenate(v) for k,v in collected.items()}

def snapshot(model,geometry,nu,out,step,device,n=10000,batch=256):
    folder=Path(out)/f'step_{step:07d}';folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(93713) # independent, identical holdout points across snapshots
    q,normal,ds=geometry.quadrature();wall=evaluate(model,q,nu,device,batch)
    coeff,forces=wall_coefficients(wall,normal,ds,nu)
    wall.update(coeff,normal_x=-normal[:,0],normal_y=-normal[:,1],area=ds)
    write_vtp(folder/'boundary_1.vtp',q,wall)
    for name in ['near','far']:
        q=geometry.volume(n,rng,name);fields=evaluate(model,q,nu,device,batch)
        write_vtp(folder/(name+'_field_1.vtp'),q,fields)
    probes=np.array([[0.,0.],[-2.,0.],[-2.,1.],[-2.,-1.]])
    values=evaluate(model,probes,nu,device,batch)
    write_vtp(folder/'leading_edge_and_freestream.vtp',probes,values)
    q=geometry.wall_layers(min(n,2048),rng);r=evaluate(model,q,nu,device,batch,full=True)
    write_vtp(folder/'holdout_wall_layers.vtp',q,r)
    a,b,c,d=geometry.bounds
    inlet=np.column_stack([np.full(512,a),np.linspace(c,d,512)])
    inf=evaluate(model,inlet,nu,device,batch)
    write_vtp(folder/'holdout_inlet.vtp',inlet,inf)
    p_inf=float(np.mean(values['p'][2:]))
    monitor=dict(step=step,**forces,p_LE=float(values['p'][0]),p_inf_probe=p_inf,p_LE_minus_p_inf=float(values['p'][0]-p_inf),
        wall_speed_rms=float(np.sqrt(np.mean(wall['u']**2+wall['v']**2))),
        wall_k_max=float(wall['k'].max()),wall_ep_tilde_max=float(wall['ep_tilde'].max()),wall_ep_physical_max=float(wall['ep'].max()),
        inlet_k_relative_rms=float(np.sqrt(np.mean((inf['k']/model.k_inf-1)**2))),
        inlet_ep_relative_rms=float(np.sqrt(np.mean((inf['ep_tilde']/model.et_inf-1)**2))),
        wall_nut_over_nu_max=float(wall['nu_t_over_nu'].max()),
        holdout_residual_rms={k:float(np.sqrt(np.mean(r[k]**2))) for k in ['continuity','momentum_x','momentum_y','k_equation','ep_equation','k_residual_raw','ep_residual_raw']})
    (folder/'metrics.json').write_text(json.dumps(monitor,indent=2))
    return monitor
