"""Reproduce the archive audit. No checkpoint execution or fitted reference truth."""
import sys, os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if (ROOT/'deps').exists(): sys.path.insert(0,str(ROOT/'deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import argparse, json, csv, hashlib, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from scipy.interpolate import PchipInterpolator
from vtp_io import read_vtp

plt.rcParams.update({'font.size':9,'axes.grid':False,'savefig.dpi':150})
NU=1/6e6; KINF=1.5e-6; EPINF=.09*KINF**2/(10*NU)

def first_zone(p):
    rows=[]; active=False
    for line in Path(p).read_text().splitlines():
        if line.lower().startswith('zone'):
            if active: break
            active=True; continue
        if active:
            try: rows.append([float(v) for v in line.split()[:2]])
            except ValueError: pass
    a=np.array(rows)
    return a[np.argsort(a[:,0])]

def polygon(n=501):
    x=.5*(1-np.cos(np.linspace(0,np.pi,n)))
    y=.594689181*(.298222773*np.sqrt(x)-.127125232*x-.357907906*x*x+.291984971*x**3-.105174606*x**4)
    y[[0,-1]]=0
    return np.column_stack([np.r_[x[::-1],x[1:-1]],np.r_[y[::-1],-y[1:-1]]])

def surface(d):
    # Archive normals point OUT OF THE FLUID and INTO THE AIRFOIL.
    nx=-d['normal_x'].astype(float); ny=-d['normal_y'].astype(float)
    tx=np.abs(ny); ty=-nx*np.sign(ny)  # both sides: LE toward TE
    shear=NU*(d['u__y']+d['v__x'])
    fx=2*NU*d['u__x']*nx+shear*ny
    fy=shear*nx+2*NU*d['v__y']*ny
    cf=2*(fx*tx+fy*ty)
    # Include the tiny modeled turbulent stress at the approximate wall only as a diagnostic.
    ratio=.09*d['k']**2/d['ep']/NU
    return dict(cp=2*d['p'],cf=cf,nx=nx,ny=ny,fx=fx,fy=fy,ratio=ratio)

def closed_forces(d, stride=1):
    # Order by perimeter angle about an interior point, then use closed trapezoidal traction.
    order=np.argsort(np.arctan2(d['y'],d['x']-.5))[::stride]
    q=np.column_stack([d['x'][order],d['y'][order]]).astype(float)
    e=np.roll(q,-1,axis=0)-q
    length=np.linalg.norm(e,axis=1)
    nds=np.column_stack([e[:,1],-e[:,0]])  # solid-outward for CCW polygon
    p=d['p'][order].astype(float); pm=.5*(p+np.roll(p,-1))
    fp=-2*np.sum(pm[:,None]*nds,axis=0)
    s=surface(d)
    f=np.column_stack([s['fx'][order],s['fy'][order]])
    fv=2*np.sum(.5*(f+np.roll(f,-1,axis=0))*length[:,None],axis=0)
    return fp,fv

def stats(a):
    a=np.asarray(a,dtype=float)
    return {'n':int(a.size),'nonfinite':int((~np.isfinite(a)).sum()),'min':float(np.nanmin(a)),
            'p01':float(np.nanquantile(a,.01)),'median':float(np.nanmedian(a)),
            'p99':float(np.nanquantile(a,.99)),'max':float(np.nanmax(a)),
            'rms':float(np.sqrt(np.nanmean(a*a)))}

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),bbox_inches='tight'); plt.close(fig)

def field(ax,d,key,norm=None,cmap='viridis',size=.9):
    v=d[key]
    sc=ax.scatter(d['x'],d['y'],c=v,s=size,cmap=cmap,norm=norm,rasterized=True,linewidths=0)
    poly=polygon(); ax.fill(poly[:,0],poly[:,1],color='white',ec='#24334a',lw=.5,zorder=5)
    ax.set_aspect('equal'); ax.set_title(key); ax.set_xlabel('x/c'); ax.set_ylabel('y/c')
    return sc

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,default=ROOT/'data/k_e_noslip')
    ap.add_argument('--out',type=Path,default=ROOT/'results'); ap.add_argument('--skip-atlas',action='store_true')
    args=ap.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=True)
    figs=out/'figures'; figs.mkdir(exist_ok=True)
    steps=sorted(args.data.glob('step_*')); all_data={}; inventory=[]; residuals=[]; metrics=[]; hashes=[]
    for folder in steps:
        step=int(folder.name.split('_')[1]); all_data[step]={}
        manifest=json.loads((folder/'manifest.json').read_text())
        for rel,info in manifest['files'].items():
            p=folder/rel; actual=hashlib.sha256(p.read_bytes()).hexdigest()
            if actual!=info['sha256']: raise ValueError('Hash mismatch '+str(p))
            hashes.append(dict(step=step,path=rel,sha256=actual))
        for p in sorted(folder.rglob('*.vtp')):
            rel=p.relative_to(folder).as_posix(); d={k:v.astype(float) for k,v in read_vtp(p).items()}
            all_data[step][rel]=d
            for key,a in d.items(): inventory.append(dict(step=step,file=rel,field=key,**stats(a)))
            for key in d:
                if not key.startswith('pred_'): continue
                key=key[5:]; e=d['pred_'+key]-d['true_'+key]; w=d.get('lambda_'+key,np.ones_like(e))
                residuals.append(dict(step=step,file=rel,field=key,rms=float(np.sqrt(np.mean(e**2))),
                    p99=float(np.quantile(abs(e),.99)),max=float(abs(e).max()),
                    weighted_integral=float(np.sum(e**2*w*d['area'])),
                    relative_rms=float(np.sqrt(np.mean(e**2))/np.max(abs(d['true_'+key]))) if np.max(abs(d['true_'+key]))>0 else None))
        d=all_data[step]['inferencers/boundary_1.vtp']; s=surface(d)
        fp,fv=closed_forces(d); fp2,fv2=closed_forces(d,2)
        area=d['area']; mc_p=np.array([np.sum(s['cp']*d['normal_x']*area),np.sum(s['cp']*d['normal_y']*area)])
        mc_v=2*np.array([np.sum(s['fx']*area),np.sum(s['fy']*area)])
        near=all_data[step]['inferencers/near_field_1.vtp']; far=all_data[step]['inferencers/far_field_1.vtp']
        le=int(np.argmin(d['x'])); upstream=(far['x'] < -2)&(abs(far['y'])>.5)
        pref=float(np.median(far['p'][upstream]))
        inlet=all_data[step]['constraints/inlet.vtp']; nb=all_data[step]['constraints/interior_near_field.vtp']
        # Side-to-side Cp comparison on common x support; no extrapolation.
        curve=[]
        for side in [d['y']>=0,d['y']<0]:
            ix=np.argsort(d['x'][side]); curve.append((d['x'][side][ix],s['cp'][side][ix]))
        xx=np.linspace(max(z[0][0] for z in curve),min(z[0][-1] for z in curve),1000)
        delta=np.interp(xx,*curve[0])-np.interp(xx,*curve[1])
        ratio=.09*near['k']**2/near['ep']/NU
        m=dict(step=step,p_le_sample=float(d['p'][le]),x_le_sample=float(d['x'][le]),y_le_sample=float(d['y'][le]),
            p_wall_max=float(d['p'].max()),cp_le_sample=float(s['cp'][le]),p_inf_estimate=pref,
            p_le_relative=float(d['p'][le]-pref),le_deficit_pct=float(100*(.5-d['p'][le])/.5),
            cl_pressure=float(fp[1]),cl_viscous=float(fv[1]),cl_total=float(fp[1]+fv[1]),
            cd_pressure=float(fp[0]),cd_viscous=float(fv[0]),cd_total=float(fp[0]+fv[0]),
            cl_mc=float(mc_p[1]+mc_v[1]),cd_mc=float(mc_p[0]+mc_v[0]),
            cl_decimation_delta=float(abs((fp+fv-fp2-fv2)[1])),cd_decimation_delta=float(abs((fp+fv-fp2-fv2)[0])),
            normal_quadrature_x=float(np.sum(d['normal_x']*area)),normal_quadrature_y=float(np.sum(d['normal_y']*area)),
            wall_speed_rms=float(np.sqrt(np.mean(d['u']**2+d['v']**2))),wall_speed_max=float(np.hypot(d['u'],d['v']).max()),
            cf_median=float(np.median(s['cf'])),cf_max=float(s['cf'].max()),cf_min=float(s['cf'].min()),
            cf_negative_fraction=float(np.mean(s['cf']<0)),wall_k_max=float(d['k'].max()),wall_ep_max=float(d['ep'].max()),
            k_near_median=float(np.median(near['k'])),k_near_max=float(near['k'].max()),
            ep_near_median=float(np.median(near['ep'])),ep_near_max=float(near['ep'].max()),
            nut_over_nu_near_median=float(np.median(ratio)),nut_over_nu_near_max=float(ratio.max()),
            inlet_k_median_ratio=float(np.median(inlet['k'])/KINF),inlet_ep_median_ratio=float(np.median(inlet['ep'])/EPINF),
            cp_symmetry_rms=float(np.sqrt(np.mean(delta**2))),
            near_sdf_min=float(nb['sdf'].min()),near_sdf_lt_1e5=int(np.sum(nb['sdf']<1e-5)),
            near_sdf_lt_1e4=int(np.sum(nb['sdf']<1e-4)),near_sdf_lt_1e3=int(np.sum(nb['sdf']<1e-3)))
        metrics.append(m)
        np.savetxt(out/f'surface_{step}.csv',np.column_stack([d['x'],d['y'],s['cp'],s['cf'],d['p'],d['k'],d['ep'],d['u'],d['v']]),delimiter=',',header='x,y,Cp,Cf,p,k,ep,u,v',comments='')
        print(json.dumps(m),flush=True)
    for name,rows in [('inventory',inventory),('residuals',residuals),('metrics',metrics),('hashes',hashes)]:
        (out/(name+'.json')).write_text(json.dumps(rows,indent=2),encoding='utf8')
        with (out/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Temporal comparisons use identical limits and un-interpolated point samples.
    for region in ['near_field','far_field']:
        dd=[all_data[s]['inferencers/'+region+'_1.vtp'] for s in all_data]
        for d in dd:
            d['speed']=np.hypot(d['u'],d['v']); d['nu_t/nu']=.09*d['k']**2/d['ep']/NU
        for key in ['u','v','p','k','ep','nu_t/nu','speed']:
            lo=min(d[key].min() for d in dd); hi=max(d[key].max() for d in dd)
            norm=LogNorm(max(lo,1e-18),hi) if key in ['k','ep','nu_t/nu'] else Normalize(lo,hi)
            fig,axs=plt.subplots(1,len(dd),figsize=(19,3.5),layout='constrained')
            for ax,d,step in zip(axs,dd,all_data):
                sc=field(ax,d,key,norm,'coolwarm' if key in ['v','p'] else 'viridis')
                ax.set_title(f'{step:,} steps')
            fig.colorbar(sc,ax=axs,label=key,shrink=.8)
            fig.suptitle(region+' | '+key+' | common scale; raw samples')
            save(fig,figs,region+'_'+key.replace('/','_'))
    latest=steps[-1]; last=int(latest.name.split('_')[1]); nd=all_data[last]['inferencers/near_field_1.vtp']
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,key in zip(axs.flat,['u','v','p','k','ep','nu_t/nu']):
        norm=LogNorm(nd[key].min(),nd[key].max()) if key in ['k','ep','nu_t/nu'] else None
        sc=field(ax,nd,key,norm,'coolwarm' if key in ['v','p'] else 'viridis',2)
        fig.colorbar(sc,ax=ax,shrink=.8)
    fig.suptitle(f'Near field at {last:,} steps | no interpolation')
    save(fig,figs,'overview_latest')
    cp_ref=first_zone(ROOT/'references/n0012cp_cfl3d_sa.dat'); cf_ref=first_zone(ROOT/'references/n0012cf_cfl3d_sa.dat')
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained'); colors=plt.cm.viridis(np.linspace(.1,.9,len(steps)))
    for color,(step,data) in zip(colors,all_data.items()):
        d=data['inferencers/boundary_1.vtp']; s=surface(d)
        for side,ls in [(d['y']>=0,'-'),(d['y']<0,'--')]:
            ix=np.argsort(d['x'][side]); xx=d['x'][side][ix]
            for ax,key in [(axs[0,0],'cp'),(axs[0,1],'cf'),(axs[1,0],'cp')]:
                ax.plot(xx,s[key][side][ix],ls,c=color,lw=1,label=f'{step//1000}k' if ls=='-' else None)
    axs[0,0].plot(cp_ref[:,0],cp_ref[:,1],c='black',lw=1.5,label='NASA CFL3D SA, M=0.15')
    axs[0,1].plot(cf_ref[:,0],cf_ref[:,1],c='black',lw=1.5,label='NASA CFL3D SA')
    axs[1,0].axhline(1,c='black',ls=':',label='Cp stagnation ~ 1')
    axs[0,0].set(ylabel='Cp',title='Surface pressure: solid upper / dashed lower',xlim=(0,1));axs[0,0].invert_yaxis()
    axs[0,1].set(ylabel='Cf (signed LE -> TE)',title='Skin friction; molecular wall stress',xlim=(0,1));axs[0,1].set_yscale('symlog',linthresh=1e-5)
    axs[1,0].set(xlim=(0,.025),ylabel='Cp',title='Leading-edge pressure');axs[1,0].invert_yaxis()
    st=[m['step'] for m in metrics]
    axs[1,1].plot(st,[m['cl_total'] for m in metrics],'o-',label='CL closed contour')
    axs[1,1].plot(st,[m['cl_mc'] for m in metrics],'x--',label='CL random area quadrature')
    axs[1,1].axhline(0,c='black',lw=.8);axs[1,1].set(title='Lift and integration sensitivity',xlabel='Training step',ylabel='CL')
    for ax in axs.flat: ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in [axs[0,0],axs[0,1],axs[1,0]]:ax.set_xlabel('x/c')
    save(fig,figs,'aerodynamics')
    # Stagnation, inlet collapse and wall diagnostics.
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,key,label in [(axs[0,0],'p_le_sample','p at closest LE sample'),(axs[0,1],'inlet_k_median_ratio','median inlet k / target'),(axs[0,2],'inlet_ep_median_ratio','median inlet ep / target'),(axs[1,0],'nut_over_nu_near_median','median near nu_t/nu'),(axs[1,1],'wall_speed_rms','wall velocity RMS'),(axs[1,2],'cd_total','CD closed contour')]:
        ax.plot(st,[m[key] for m in metrics],'o-');ax.set(title=label,xlabel='Training step');ax.grid(alpha=.2)
    axs[0,0].axhline(.5,c='black',ls='--',label='Bernoulli scale');axs[0,0].legend()
    for ax in axs[0,1:]:ax.axhline(1,c='black',ls='--');ax.set_ylim(0,1.05)
    axs[1,0].set_yscale('log');axs[1,2].axhline(.00819,c='black',ls='--',label='NASA SA representative');axs[1,2].legend(fontsize=7)
    save(fig,figs,'convergence')
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    d=all_data[last]['inferencers/boundary_1.vtp'];s=surface(d)
    for ax,key in zip(axs.flat,['u','v','k','ep','u__y','v__x']):
        for side,color in [(d['y']>=0,'tab:blue'),(d['y']<0,'tab:orange')]:
            ix=np.argsort(d['x'][side]);ax.plot(d['x'][side][ix],d[key][side][ix],c=color)
        ax.set(title=key,xlabel='x/c');ax.grid(alpha=.2)
        if key in ['k','ep']:ax.set_yscale('log')
    fig.suptitle('Surface fields and derivatives | blue upper, orange lower')
    save(fig,figs,'wall_latest')
    # Reconstruct actual equation terms to expose scale collapse, without replacing saved AD residuals.
    d=all_data[last]['constraints/interior_near_field.vtp'];nt=.09*d['k']**2/d['ep'];pk=nt*(2*d['u__x']**2+2*d['v__y']**2+(d['u__y']+d['v__x'])**2)
    nx=nt*(2*d['k__x']/d['k']-d['ep__x']/d['ep']);ny=nt*(2*d['k__y']/d['k']-d['ep__y']/d['ep'])
    terms={'k_advection':d['u']*d['k__x']+d['v']*d['k__y'],
           'k_diffusion':(NU+nt)*(d['k__x__x']+d['k__y__y'])+nx*d['k__x']+ny*d['k__y'],
           'production':pk,'ep':d['ep'],
           'ep_advection':d['u']*d['ep__x']+d['v']*d['ep__y'],
           'ep_diffusion':(NU+nt/1.3)*(d['ep__x__x']+d['ep__y__y'])+(nx*d['ep__x']+ny*d['ep__y'])/1.3,
           'ep_production':1.44*pk*d['ep']/d['k'],'ep_destruction':1.92*d['ep']**2/d['k']}
    (out/'term_statistics.json').write_text(json.dumps({k:stats(v) for k,v in terms.items()},indent=2))
    reconstructed_k=(terms['k_advection']-terms['k_diffusion']-pk+d['ep'])/1e-3
    reconstructed_ep=(terms['ep_advection']-terms['ep_diffusion']-terms['ep_production']+terms['ep_destruction'])/1e-2
    (out/'pde_reconstruction.json').write_text(json.dumps({'k_max_abs_difference':float(np.max(abs(reconstructed_k-d['k_equation']))),'ep_max_abs_difference':float(np.max(abs(reconstructed_ep-d['ep_equation'])))},indent=2))
    # All nonredundant numeric arrays, including every derivative/residual, are in the atlas.
    # Constants and duplicate true/pred/lambda arrays remain fully inventoried in CSV/JSON.
    atlas=[]
    if not args.skip_atlas:
        for step,data in all_data.items():
            for rel,d in data.items():
                keys=[k for k,v in d.items() if k not in ['x','y','z','alpha','camber','thickness','area','nu_t/nu','speed'] and not k.startswith(('true_','pred_','lambda_'))]
                for page,start in enumerate(range(0,len(keys),12)):
                    subset=keys[start:start+12]; fig,axs=plt.subplots(math.ceil(len(subset)/3),3,figsize=(14,3*math.ceil(len(subset)/3)),squeeze=False,layout='constrained')
                    for ax,key in zip(axs.flat,subset):
                        a=d[key];norm=LogNorm(a.min(),a.max()) if a.min()>0 and a.max()/a.min()>100 else None
                        if 'boundary' in rel or any(z in rel for z in ['airfoil','inlet','outlet','top_bot']):
                            coord='y' if any(z in rel for z in ['inlet','outlet']) else 'x';ax.scatter(d[coord],a,s=2);ax.set(xlabel=coord,title=key)
                            if norm is not None:ax.set_yscale('log')
                        else:
                            sc=field(ax,d,key,norm,size=1);fig.colorbar(sc,ax=ax,shrink=.8)
                    for ax in list(axs.flat)[len(subset):]:ax.set_visible(False)
                    title=f'{step} | {rel} | page {page+1}'; fig.suptitle(title)
                    name=f'atlas_{step}_{rel.replace("/","_").replace(".vtp","")}_{page+1}'
                    save(fig,figs,name);atlas.append((title,name+'.png'))
        (out/'atlas.html').write_text('<!doctype html><meta charset="utf-8"><title>All-field atlas</title><style>body{font:16px system-ui;max-width:1500px;margin:auto}img{width:100%}summary{padding:16px;cursor:pointer}</style><h1>All-field atlas</h1><p>Every unique field in all 55 VTP files. Constants, duplicate targets/predictions and lambda weights are tabulated in inventory.csv.</p>'+''.join(f'<details><summary>{title}</summary><img loading="lazy" src="figures/{name}"></details>' for title,name in atlas),encoding='utf8')
    print('DONE',len(inventory),'array records;',len(hashes),'verified hashes;',len(atlas),'atlas pages')

if __name__=='__main__':main()
