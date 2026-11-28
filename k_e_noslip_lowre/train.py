"""Standalone, auditable low-Re replacement; does not load old PhysicsNeMo weights."""
import argparse, hashlib, json, math, platform, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml
from geometry import Geometry
from model import FlowModel,freestream,physics,boundary_losses
from diagnostics import snapshot

def tensors(q,device,dtype):return torch.as_tensor(q,device=device,dtype=dtype).requires_grad_(True)

def samples(geo,cfg,epoch):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch]));n=cfg['batch_boundary']
    a,b,c,d=geo.bounds
    wall,_=geo.boundary(n,rng)
    inlet=np.column_stack([np.full(n,a),rng.uniform(c,d,n)])
    outer=np.column_stack([rng.uniform(a,b,n),rng.choice([c,d],n)])
    outlet=np.column_stack([np.full(n,b),rng.uniform(c,d,n)])
    regions={'near':geo.volume(cfg['batch_near'],rng),'far':geo.volume(cfg['batch_far'],rng,'far'),
        'layers':geo.wall_layers(cfg['batch_layers'],rng,cfg['dmin'],cfg['dmax']),
        'leading':geo.leading_edge(cfg['batch_leading'],rng),'wake':geo.wake(cfg['batch_wake'],rng)}
    return (wall,inlet,outer,outlet),regions

def train(cfg,out,resume=None,smoke=False):
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    if cfg['device']=='auto':cfg['device']='cuda' if torch.cuda.is_available() else 'cpu'
    device=torch.device(cfg['device']);dtype=getattr(torch,cfg['dtype'])
    if dtype not in [torch.float32,torch.float64]:raise ValueError('Use float32 or float64; no AMP for stiff high-order residuals')
    if cfg['reynolds']<=0:raise ValueError('Re must be positive')
    out=Path(out)
    if out.exists() and any(out.iterdir()) and resume is None:raise ValueError('Use a new output folder or --resume')
    out.mkdir(parents=True,exist_ok=True)
    geo=Geometry(cfg['resolution'],cfg['bounds']);nu=1/cfg['reynolds'];ki,ei=freestream(nu,cfg['intensity'],cfg['viscosity_ratio'])
    model=FlowModel(ki,ei,cfg['width'],cfg['layers']).to(device=device,dtype=dtype)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']);start=0
    if resume:
        state=torch.load(resume,map_location=device,weights_only=True)
        # Check physical/architecture/sampling settings, while allowing a new max_steps/record interval.
        for key in ['reynolds','intensity','viscosity_ratio','width','layers','dtype','resolution','bounds','seed','dmin','dmax']:
            if state['config'][key]!=cfg[key]:raise ValueError('Resume configuration mismatch: '+key)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step']
    provenance={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
                'device':str(device),'k_inf':ki,'ep_tilde_inf':ei,'config':cfg,
                'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (out/'run_manifest.json').write_text(json.dumps(provenance,indent=2));(out/'config.yaml').write_text(yaml.safe_dump(cfg))
    data=None;epoch=-1;since=time.perf_counter()
    with (out/'losses.jsonl').open('a',encoding='utf8') as log:
        for step in range(start+1,cfg['max_steps']+1):
            new_epoch=(step-1)//cfg['resample_every']
            if new_epoch!=epoch:data=samples(geo,cfg,new_epoch);epoch=new_epoch
            optimizer.zero_grad(set_to_none=True)
            bc=[tensors(a,device,dtype) for a in data[0]]
            losses=boundary_losses(model,*bc)
            bc_loss=sum((cfg['velocity_bc_weight'] if 'velocity' in k else 1.)*v for k,v in losses.items())
            if not torch.isfinite(bc_loss):raise FloatingPointError('Nonfinite boundary loss')
            bc_loss.backward();records={k:float(v.detach()) for k,v in losses.items()}
            # BCs are fully active from step one; only the PDE multiplier ramps.
            ramp=.1+.9*min(step/cfg['pde_ramp_steps'],1.)
            for name,q in data[1].items():
                xy=tensors(q,device,dtype);res,diag=physics(model,xy,nu)
                part=sum(r.square().mean() for r in res.values())*ramp/len(data[1])
                if not torch.isfinite(part):raise FloatingPointError('Nonfinite PDE loss in '+name)
                part.backward()
                for key,r in res.items():records[name+'/'+key]=float(r.detach().square().mean())
                for key in ['k_residual_raw','ep_residual_raw']:
                    records[name+'/'+key+'_rms']=float(diag[key].detach().square().mean().sqrt())
            # Report each subnetwork's gradient BEFORE clipping, so imbalance is visible.
            for name in ['flow','pressure','tke','dissipation']:
                pars=list(getattr(model,name).parameters())
                records['grad_norm/'+name]=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in pars if p.grad is not None))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
            # A floor avoids a practically frozen tail; absolute-step schedule is resume invariant.
            lr=max(cfg['lr_min'],cfg['lr']*.95**(step/8000))
            for g in optimizer.param_groups:g['lr']=lr
            optimizer.step()
            if step%cfg['log_every']==0 or step==start+1:
                records.update(step=step,lr=lr,pde_multiplier=ramp,grad_norm=float(norm),seconds=time.perf_counter()-since)
                log.write(json.dumps(records)+'\n');log.flush();print(json.dumps({'step':step,'boundary_loss':float(bc_loss.detach()),'grad_norm':float(norm),'lr':lr}),flush=True)
            if step%cfg['record_every']==0 or step==cfg['max_steps']:
                state={'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg}
                torch.save(state,out/'checkpoint_latest.pt')
                # Independent spatial validation, full physical/modified epsilon and raw residuals.
                metrics=snapshot(model,geo,nu,out,step,device,n=cfg['inference_points'],batch=cfg['inference_batch'])
                print(json.dumps(metrics),flush=True)
    return model

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    ap.add_argument('--out',required=True);ap.add_argument('--resume');ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();cfg=yaml.safe_load(args.config.read_text())
    if args.smoke:
        cfg.update(width=12,layers=2,resolution=51,max_steps=2,batch_boundary=16,batch_near=8,batch_far=8,
            batch_layers=8,batch_leading=8,batch_wake=8,inference_points=16,inference_batch=16,log_every=1,record_every=2,device='cpu')
    train(cfg,args.out,args.resume,args.smoke)

if __name__=='__main__':main()
