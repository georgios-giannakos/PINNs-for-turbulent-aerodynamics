import sys,os,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import surface,first_zone,closed_forces,save,NU
from vtp_io import read_vtp
out=ROOT/'results';figs=out/'figures';rows=[]
cp=first_zone(ROOT/'references/n0012cp_cfl3d_sa.dat');cf=first_zone(ROOT/'references/n0012cf_cfl3d_sa.dat')
fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained')
for folder in sorted((ROOT/'data/k_e_noslip').glob('step_*')):
    step=int(folder.name.split('_')[1]);d=read_vtp(folder/'inferencers/boundary_1.vtp');d={k:v.astype(float) for k,v in d.items()};s=surface(d)
    profiles={}
    for side,label in [(d['y']>=0,'upper'),(d['y']<0,'lower')]:
        ix=np.argsort(d['x'][side]);xx=d['x'][side][ix];prof={}
        for key in ['cp','cf','p']:
            val=d['p'][side][ix] if key=='p' else s[key][side][ix]
            prof[key]=np.interp([.01,.1,.5,.9],xx,val).tolist()
        profiles[label]=prof
    # Compare both sides to the representative symmetric SA solution on common x, avoiding endpoints.
    grid=np.linspace(.01,.99,2000)
    cp_unique=np.unique(cp[:,0]);cp_mean=np.array([cp[cp[:,0]==x,1].mean() for x in cp_unique])
    ratios=[];cp_errors=[]
    for side in [d['y']>=0,d['y']<0]:
        ix=np.argsort(d['x'][side]);x=d['x'][side][ix]
        ratios.extend((np.interp(grid,x,s['cf'][side][ix])/np.interp(grid,cf[:,0],cf[:,1])).tolist())
        cp_errors.extend((np.interp(grid,x,s['cp'][side][ix])-np.interp(grid,cp_unique,cp_mean)).tolist())
    near={k:v.astype(float) for k,v in read_vtp(folder/'constraints/interior_near_field.vtp').items()}
    far=read_vtp(folder/'inferencers/far_field_1.vtp')
    # LE estimate at x=y=0 by interpolation of the two nearest opposite-side samples in y.
    upper=np.where(d['y']>=0)[0];lower=np.where(d['y']<0)[0]
    iu=upper[np.argmin(d['x'][upper])];il=lower[np.argmin(d['x'][lower])]
    p0=float(np.interp(0,[d['y'][il],d['y'][iu]],[d['p'][il],d['p'][iu]]))
    nclose=near['sdf']<.001
    residual_by_distance=[]
    for low,high in [(0,1e-4),(1e-4,1e-3),(1e-3,1e-2),(1e-2,1)]:
        sel=(near['sdf']>=low)&(near['sdf']<high)
        residual_by_distance.append({'dmin':low,'dmax':high,'n':int(sel.sum()),'continuity_rms':float(np.sqrt(np.mean(near['continuity'][sel]**2))) if sel.any() else None,'momentum_x_rms':float(np.sqrt(np.mean(near['momentum_x'][sel]**2))) if sel.any() else None})
    row=dict(step=step,profiles=profiles,cf_ratio_to_SA_median=float(np.median(ratios)),cp_rmse_to_SA=float(np.sqrt(np.mean(np.array(cp_errors)**2))),
        p_le_two_side_linear=p0,le_pair_x=[float(d['x'][il]),float(d['x'][iu])],
        residual_by_distance=residual_by_distance,
        wake_min_u=float(far['u'][far['x']>1].min()),
        nt_wall_max=float(s['ratio'].max()))
    rows.append(row)
    if step==400000:
        for ax,key in zip(axs[0],['continuity','momentum_x','k_equation']):
            ax.scatter(near['sdf'],abs(near[key]),s=2,alpha=.5);ax.set(xscale='log',yscale='log',xlabel='stored SDF',ylabel='absolute residual',title=key)
        rr=json.loads((out/'residuals.json').read_text());r=[r for r in rr if r['step']==400000 and 'init' not in r['field']]
        labels=[v['file'].split('/')[1].replace('.vtp','')+':'+v['field'] for v in r]
        axs[1,0].barh(labels,[v['weighted_integral'] for v in r]);axs[1,0].set(xscale='log',title='Reconstructed weighted constraint losses');axs[1,0].tick_params(axis='y',labelsize=6)
        inlet=read_vtp(folder/'constraints/inlet.vtp');ix=np.argsort(inlet['y'])
        axs[1,1].plot(inlet['y'][ix],inlet['k'][ix]/inlet['true_k'][ix],label='k / target')
        axs[1,1].plot(inlet['y'][ix],inlet['ep'][ix]/inlet['true_ep'][ix],label='ep / target');axs[1,1].axhline(1,c='black',ls='--');axs[1,1].set(title='Turbulence inlet fails',xlabel='y/c');axs[1,1].legend()
        axs[1,2].plot(d['x'],np.sqrt(abs(s['cf'])/2)/NU*1e-6,'.',label='at d/c=1e-6')
        axs[1,2].set(title='y+ inferred from current wall shear',xlabel='x/c',ylabel='y+');axs[1,2].legend()
fig.suptitle('Why a small training loss is misleading | 400,000 steps')
save(fig,figs,'loss_diagnosis')
(out/'additional_metrics.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows[-1],indent=2))
