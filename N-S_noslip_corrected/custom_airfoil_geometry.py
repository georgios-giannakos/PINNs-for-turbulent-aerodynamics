# SPDX-License-Identifier: Apache-2.0
"""Shared TMR polygon, Warp SDF and arc-length/rejection sampling.

Positive signed distance in fluid; normals point out of the fluid.
Thickness is the nominal NACA parameter (0.12 gives the TMR section).
Other camber/thickness values are generalized profiles, not TMR benchmarks.
"""
import numpy as np
import warp as wp

wp.init()


@wp.func
def vertex(j: int, n: int, p: wp.vec3, c: float, loc: float) -> wp.vec2:
    idx = j
    side = float(-1.0)
    if j < n:
        idx = n-1-j
        side = 1.0
    else:
        idx = j-n+1
    x = 0.5*(1.0-wp.cos(3.141592653589793*float(idx)/float(n-1)))
    t = 0.594689181*(p[2]/0.12)*(0.298222773*wp.sqrt(wp.max(x,0.0))
        -0.127125232*x-0.357907906*x*x+0.291984971*x*x*x-0.105174606*x*x*x*x)
    if idx == 0 or idx == n-1:
        t = 0.0
    m = p[1]
    yc = float(0.0)
    slope = float(0.0)
    if x < loc:
        yc = m*(2.0*loc*x-x*x)/(loc*loc)
        slope = 2.0*m*(loc-x)/(loc*loc)
    else:
        d = 1.0-loc
        yc = m*(1.0-2.0*loc+2.0*loc*x-x*x)/(d*d)
        slope = 2.0*m*(loc-x)/(d*d)
    theta = wp.atan(slope)
    a = x-side*t*wp.sin(theta)
    b = yc+side*t*wp.cos(theta)
    ca = wp.cos(p[0])
    sa = wp.sin(p[0])
    return c*wp.vec2(ca*a-sa*b,sa*a+ca*b)


@wp.kernel
def vertices_kernel(params: wp.array(dtype=wp.vec3), n: int, ns: int,
                    c: float, loc: float, verts: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    verts[i] = vertex(i%ns,n,params[i//ns],c,loc)


@wp.kernel
def measures_kernel(verts: wp.array(dtype=wp.vec2), ns: int, lim: wp.vec4,
                    measures: wp.array(dtype=wp.vec2), valid: wp.array(dtype=wp.int32)):
    i = wp.tid()
    area = float(0.0)
    length = float(0.0)
    ok = int(1)
    for j in range(ns):
        a = verts[i*ns+j]
        b = verts[i*ns+(j+1)%ns]
        ell = wp.length(b-a)
        if ell <= 0.0 or a[0] <= lim[0] or a[0] >= lim[1] or a[1] <= lim[2] or a[1] >= lim[3]:
            ok = 0
        area += a[0]*b[1]-a[1]*b[0]
        length += ell
    if area <= 0.0 or length <= 0.0:
        ok = 0
    measures[i] = wp.vec2(0.5*area,length)
    valid[i] = ok


@wp.func
def fluid_distance(q: wp.vec2, verts: wp.array(dtype=wp.vec2),
                   off: int, ns: int, lim: wp.vec4) -> wp.vec3:
    best = float(1.0e20)
    residual = wp.vec2(0.0,0.0)
    fallback = wp.vec2(0.0,0.0)
    inside = int(0)
    for j in range(ns):
        a = verts[off+j]
        b = verts[off+(j+1)%ns]
        e = b-a
        l2 = wp.dot(e,e)
        if l2 > 0.0:
            t = wp.clamp(wp.dot(q-a,e)/l2,0.0,1.0)
            rr = q-a-t*e
            d2 = wp.dot(rr,rr)
            if d2 < best:
                best = d2
                residual = rr
                fallback = wp.vec2(e[1],-e[0])/wp.sqrt(l2)
        if (a[1]>q[1] and b[1]<=q[1]) or (b[1]>q[1] and a[1]<=q[1]):
            hit = a[0]+(q[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if q[0]<hit:
                inside = 1-inside
    d = wp.sqrt(wp.max(best,0.0))
    sign = float(1.0)
    if inside == 1:
        sign = -1.0
    g = fallback
    if d > 1.0e-12:
        g = sign*residual/d
    sd = sign*d
    # Preserve the original near-field convention (no artificial x boundaries).
    bottom = q[1]-lim[2]
    top = lim[3]-q[1]
    if bottom < sd and bottom <= top:
        return wp.vec3(bottom,0.0,1.0)
    if top < sd:
        return wp.vec3(top,0.0,-1.0)
    return wp.vec3(sd,g[0],g[1])


@wp.kernel
def sampling_kernel(verts: wp.array(dtype=wp.vec2), measures: wp.array(dtype=wp.vec2),
                    ns: int, varying: int, lim: wp.vec4, seed: int, wall: int,
                    out: wp.array(dtype=wp.vec4), sdf: wp.array(dtype=wp.float32),
                    weights: wp.array(dtype=wp.float32), ok: wp.array(dtype=wp.int32)):
    i = wp.tid()
    profile = int(0)
    if varying == 1:
        profile = i
    off = profile*ns
    state = wp.rand_init(seed,i)
    if wall == 1:
        length = measures[profile][1]
        s = wp.randf(state)*length
        acc = float(0.0)
        for j in range(ns):
            a = verts[off+j]
            b = verts[off+(j+1)%ns]
            e = b-a
            ell = wp.length(e)
            if ell > 0.0 and (s < acc+ell or j == ns-1):
                q = a+wp.clamp((s-acc)/ell,0.0,1.0)*e
                normal = wp.vec2(-e[1],e[0])/ell
                out[i] = wp.vec4(q[0],q[1],normal[0],normal[1])
                weights[i] = length
                ok[i] = 1
                break
            acc += ell
    else:
        # Reject spatial points, not the sampled parameter values.
        for attempt in range(1024):
            q = wp.vec2(wp.randf(state,lim[0],lim[1]),wp.randf(state,lim[2],lim[3]))
            d = fluid_distance(q,verts,off,ns,lim)
            if d[0]>0.0 and q[0]>lim[0] and q[0]<lim[1]:
                out[i] = wp.vec4(q[0],q[1],d[1],d[2])
                sdf[i] = d[0]
                weights[i] = (lim[1]-lim[0])*(lim[3]-lim[2])-measures[profile][0]
                ok[i] = 1
                break


@wp.kernel
def query_kernel(q: wp.array(dtype=wp.vec2), verts: wp.array(dtype=wp.vec2),
                 ns: int, lim: wp.vec4, result: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    result[i] = fluid_distance(q[i],verts,0,ns,lim)


class AirfoilInChannel:
    dims = ["x","y"]

    def __init__(self,x_min,x_max,y_min,y_max,params=None,include_channel_boundary=False,
                 loc_camber=0.4,chord=1.0,airfoil_resolution=501,device=None):
        if include_channel_boundary:
            raise ValueError("Use native Channel2D for the outer boundary.")
        if not (x_min<x_max and y_min<y_max and chord>0 and 0<loc_camber<1 and airfoil_resolution>=3):
            raise ValueError("Invalid geometry settings.")
        self.params=params
        self.lim=wp.vec4(float(x_min),float(x_max),float(y_min),float(y_max))
        self.n=int(airfoil_resolution)
        self.ns=2*(self.n-1)
        self.chord=float(chord)
        self.loc=float(loc_camber)
        self.device=wp.get_device(device or ("cuda:0" if wp.is_cuda_available() else "cpu"))

    def _parameters(self,n,p,rng):
        p=self.params if p is None else p
        ranges={} if p is None else {str(k):v for k,v in p.param_ranges.items()}
        cols=[]
        for name,default in (("alpha",0.0),("camber",0.0),("thickness",0.12)):
            v=ranges.get(name,default)
            if isinstance(v,tuple):
                if len(v)!=2 or v[1]<v[0]:
                    raise ValueError("Invalid parameter range.")
                z=rng.uniform(float(v[0]),float(v[1]),n)
            elif isinstance(v,np.ndarray):
                z=rng.choice(v.reshape(-1),size=n)
            else:
                z=np.full(n,float(v))
            cols.append(z)
        a=np.column_stack(cols).astype(np.float32)
        if not np.isfinite(a).all() or np.any(a[:,2]<=0):
            raise ValueError("Invalid profile parameters.")
        return a

    def _buffers(self,p):
        varying=int(not np.all(p==p[0]))
        p=p if varying else p[:1]
        n=len(p)
        v=wp.empty(n*self.ns,dtype=wp.vec2,device=self.device)
        m=wp.empty(n,dtype=wp.vec2,device=self.device)
        ok=wp.zeros(n,dtype=wp.int32,device=self.device)
        wp.launch(vertices_kernel,n*self.ns,inputs=[wp.array(p,dtype=wp.vec3,device=self.device),
                  self.n,self.ns,self.chord,self.loc,v],device=self.device)
        wp.launch(measures_kernel,n,inputs=[v,self.ns,self.lim,m,ok],device=self.device)
        if not np.all(ok.numpy()==1):
            raise ValueError("Degenerate or out-of-bounds airfoil polygon.")
        return v,m,varying

    def _sample(self,n,seed,p,wall):
        n=int(n)
        if n<1:
            raise ValueError("Positive sample count required.")
        seed=int(seed)%2147483647
        params=self._parameters(n,p,np.random.default_rng(seed))
        blocks=[]
        # Bound temporary GPU allocations for continuously varying geometries.
        chunk=2048 if not np.all(params==params[0]) else n
        for start in range(0,n,chunk):
            a=params[start:start+chunk]
            size=len(a)
            v,m,varying=self._buffers(a)
            out=wp.zeros(size,dtype=wp.vec4,device=self.device)
            sdf=wp.zeros(size,dtype=wp.float32,device=self.device)
            area=wp.zeros(size,dtype=wp.float32,device=self.device)
            ok=wp.zeros(size,dtype=wp.int32,device=self.device)
            wp.launch(sampling_kernel,size,inputs=[v,m,self.ns,varying,self.lim,
                      (seed+104729*start)%2147483647,int(wall),out,sdf,area,ok],device=self.device)
            if not np.all(ok.numpy()==1):
                raise RuntimeError("Warp sampling incomplete.")
            blocks.append((out.numpy(),sdf.numpy(),area.numpy()))
        q,d,w=(np.concatenate([b[i] for b in blocks],axis=0) for i in range(3))
        if not np.isfinite(q).all() or not np.isfinite(w).all() or np.any(w<=0):
            raise RuntimeError("Invalid geometry samples or weights.")
        if not np.allclose(np.linalg.norm(q[:,2:],axis=1),1,atol=2e-5):
            raise RuntimeError("Invalid normals or SDF gradients.")
        data=dict(x=q[:,0:1].copy(),y=q[:,1:2].copy(),area=(w/n).reshape(-1,1))
        for i,key in enumerate(("alpha","camber","thickness")):
            data[key]=params[:,i:i+1].copy()
        if wall:
            data.update(normal_x=q[:,2:3].copy(),normal_y=q[:,3:4].copy())
        else:
            if not np.isfinite(d).all() or np.any(d<=0):
                raise RuntimeError("Invalid fluid SDF.")
            data.update(sdf=d.reshape(-1,1),sdf__x=q[:,2:3].copy(),sdf__y=q[:,3:4].copy())
        return data

    def sample_boundary(self,n_points,rand_seed=0,criteria=None,parameterization=None,quasirandom=False):
        if criteria is not None or quasirandom:
            raise ValueError("Use unfiltered pseudorandom arc-length sampling.")
        return self._sample(n_points,rand_seed,parameterization,True)

    def sample_interior(self,n_points,rand_seed=0,bounds=None,criteria=None,
                        parameterization=None,compute_sdf_derivatives=False,quasirandom=False):
        if bounds is not None or criteria is not None or quasirandom:
            raise ValueError("Near rectangle uses explicit pseudorandom rejection.")
        return self._sample(n_points,rand_seed,parameterization,False)

    def polygon(self,alpha=0.0,camber=0.0,thickness=0.12):
        v,m,_=self._buffers(np.array([[alpha,camber,thickness]],dtype=np.float32))
        return v.numpy(),m.numpy()[0]

    def fluid_sdf(self,points,alpha=0.0,camber=0.0,thickness=0.12):
        v,_,_=self._buffers(np.array([[alpha,camber,thickness]],dtype=np.float32))
        q=wp.array(np.asarray(points,dtype=np.float32).reshape(-1,2),dtype=wp.vec2,device=self.device)
        out=wp.empty(len(q),dtype=wp.vec3,device=self.device)
        wp.launch(query_kernel,len(q),inputs=[q,v,self.ns,self.lim,out],device=self.device)
        return out.numpy()
