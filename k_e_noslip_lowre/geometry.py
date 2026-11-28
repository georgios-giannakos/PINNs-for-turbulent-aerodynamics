"""Fixed zero-incidence NASA TMR section; all lengths normalized by chord.

NumPy sampling is independent of autograd. No sampled SDF is used as a network
input, so spatial derivatives cannot silently lose a distance chain rule.
"""
import numpy as np

class Geometry:
    def __init__(self, resolution=1001, bounds=(-2.25,5.25,-2.5,2.5)):
        if resolution<51: raise ValueError('Use at least 51 points per side')
        self.bounds=tuple(bounds)
        x=.5*(1-np.cos(np.linspace(0,np.pi,resolution)))
        y=.594689181*(.298222773*np.sqrt(x)-.127125232*x-.357907906*x*x+.291984971*x**3-.105174606*x**4)
        y[[0,-1]]=0
        self.x=x;self.y=y
        self.vertices=np.column_stack([np.r_[x[::-1],x[1:-1]],np.r_[y[::-1],-y[1:-1]]])
        self.edge=np.roll(self.vertices,-1,axis=0)-self.vertices
        self.ds=np.linalg.norm(self.edge,axis=1)
        self.normal=np.column_stack([self.edge[:,1],-self.edge[:,0]])/self.ds[:,None] # solid -> fluid
        self.cumulative=np.r_[0,np.cumsum(self.ds)]
    def fluid(self,q):
        x,y=q.T
        return (x<0)|(x>1)|(abs(y)>np.interp(x,self.x,self.y))
    def inside_box(self,q):
        a,b,c,d=self.bounds
        return (q[:,0]>a)&(q[:,0]<b)&(q[:,1]>c)&(q[:,1]<d)
    def boundary(self,n,rng):
        # Stratified arclength; independent within each bin, random cyclic shift.
        s=(np.arange(n)+rng.random(n))/n*self.cumulative[-1]
        j=np.minimum(np.searchsorted(self.cumulative,s,side='right')-1,len(self.ds)-1)
        q=self.vertices[j]+((s-self.cumulative[j])/self.ds[j])[:,None]*self.edge[j]
        return q,self.normal[j]
    def quadrature(self):
        # Exactly closed polygon and exactly cancelling integrated normals.
        return self.vertices+.5*self.edge,self.normal.copy(),self.ds.copy()
    def volume(self,n,rng,region='near'):
        bounds=(-.32,1.28,-.5,.5) if region=='near' else self.bounds
        out=[];count=0
        while count<n:
            q=rng.uniform([bounds[0],bounds[2]],[bounds[1],bounds[3]],size=(max(64,2*(n-count)),2))
            keep=self.fluid(q)
            if region=='far':keep&=(q[:,0]<-.32)|(q[:,0]>1.28)|(abs(q[:,1])>.5)
            out.append(q[keep]);count+=keep.sum()
        return np.concatenate(out)[:n]
    def wall_layers(self,n,rng,dmin=1e-6,dmax=.02):
        if not 0<dmin<dmax:raise ValueError('Invalid wall-distance range')
        out=[];count=0
        while count<n:
            size=max(64,2*(n-count));q,normal=self.boundary(size,rng)
            d=np.exp(rng.uniform(np.log(dmin),np.log(dmax),size))
            q=q+d[:,None]*normal;keep=self.fluid(q)&self.inside_box(q)
            out.append(q[keep]);count+=keep.sum()
        # Deliberate equal importance in log-distance; NOT a physical volume quadrature.
        return np.concatenate(out)[:n]
    def leading_edge(self,n,rng):
        out=[];count=0
        while count<n:
            q=rng.uniform([-.04,-.04],[.04,.04],(max(64,2*(n-count)),2))
            q=q[self.fluid(q)];out.append(q);count+=len(q)
        return np.concatenate(out)[:n]
    def wake(self,n,rng):
        return np.column_stack([rng.uniform(1.0001,min(3,self.bounds[1]),n),rng.uniform(-.04,.04,n)])

