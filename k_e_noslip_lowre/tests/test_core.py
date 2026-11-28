"""Scientific regression tests; run: python -m unittest discover -s tests -v."""
import sys,unittest,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from geometry import Geometry
from model import FlowModel,freestream,physics
from diagnostics import wall_coefficients,write_vtp

class CoreTests(unittest.TestCase):
    def setUp(self):torch.set_default_dtype(torch.float64)
    def test_closed_quadrature_and_pressure_gauge(self):
        g=Geometry(501);q,n,ds=g.quadrature()
        np.testing.assert_allclose((n*ds[:,None]).sum(0),0,atol=1e-14)
        zero=np.zeros(len(q));f={k:zero.copy() for k in ['p','u__x','u__y','v__x','v__y']}
        f['p']=q[:,0]+23
        _,c=wall_coefficients(f,n,ds,1e-6)
        p=g.vertices;area=.5*np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))
        self.assertAlmostEqual(c['CD'],-2*area,places=12);self.assertAlmostEqual(c['CL'],0,places=12)
    def test_skin_friction_symmetric_stress(self):
        f={'p':np.array([0.]),'u__x':np.array([0.]),'u__y':np.array([4.]),'v__x':np.array([3.]),'v__y':np.array([0.])}
        c,_=wall_coefficients(f,np.array([[0.,1.]]),np.ones(1),.2)
        self.assertAlmostEqual(c['Cf'][0],2*.2*(4+3))
    def test_freestream_viscosity_ratio(self):
        for ratio in [1,10,100]:
            k,e=freestream(1/6e6,viscosity_ratio=ratio);rt=k*k/(e/6e6)
            self.assertAlmostEqual(.09*math.exp(-3.4/(1+rt/50)**2)*rt,ratio,places=10)
    def test_positive_map_and_initial_scale(self):
        ki,ei=freestream(1/6e6);m=FlowModel(ki,ei,8,2)
        for part in [m.tke,m.dissipation]:
            for p in part.parameters():torch.nn.init.zeros_(p)
        q=torch.tensor([[.2,.1],[.3,-.2]],requires_grad=True);f=m(q)
        torch.testing.assert_close(f['k'],torch.full((2,1),ki),rtol=1e-11,atol=1e-20)
        torch.testing.assert_close(f['ep_tilde'],torch.full((2,1),ei),rtol=1e-11,atol=1e-20)
        self.assertTrue(all(torch.isfinite(v).all() for v in f.values()))
    def test_sampling_fluid_and_reproducible(self):
        g=Geometry(101);a=g.wall_layers(2000,np.random.default_rng(9));b=g.wall_layers(2000,np.random.default_rng(9))
        np.testing.assert_array_equal(a,b);self.assertTrue(g.fluid(a).all())
        for mode in ['near','far']:
            q=g.volume(2000,np.random.default_rng(17),mode);self.assertTrue(g.fluid(q).all())
            if mode=='far':self.assertTrue(((q[:,0]<-.32)|(q[:,0]>1.28)|(abs(q[:,1])>.5)).all())
    def test_manufactured_equations_against_symbolic_derivatives(self):
        import sympy as s
        x,y=s.symbols('x y');u=1+x*x+s.Rational(3,10)*x*y+s.Rational(1,10)*y**3
        v=s.Rational(1,5)*x**3-s.Rational(2,5)*x*y+s.Rational(1,5)*y*y
        p=s.Rational(1,10)*x*x+y;q=s.Rational(1,5)+s.Rational(3,100)*x+s.Rational(1,50)*y+s.Rational(1,100)*x*x
        e=s.Rational(1,25)+s.Rational(3,1000)*x*x+s.Rational(1,500)*y*y
        nu=.01;k=q*q;rt=k*k/(nu*(e+1e-20));nt=.09*s.exp(-3.4/(1+rt/50)**2)*k*k/(e+1e-20)
        velocity=[u,v];coords=[x,y];S=s.Matrix(2,2,lambda i,j:(s.diff(velocity[i],coords[j])+s.diff(velocity[j],coords[i]))/2)
        P=2*nt*sum(S[i,j]**2 for i in range(2) for j in range(2))
        D=2*nu*sum(s.diff(q,z)**2 for z in coords)
        E=2*nu*nt*sum(s.diff(velocity[i],coords[j],coords[l])**2 for i in range(2) for j in range(2) for l in range(2))
        mom=[sum(velocity[j]*s.diff(velocity[i],coords[j]) for j in range(2))+s.diff(p,coords[i])+s.Rational(2,3)*s.diff(k,coords[i])-sum(s.diff(2*(nu+nt)*S[i,j],coords[j]) for j in range(2)) for i in range(2)]
        rk=sum(velocity[j]*s.diff(k,coords[j])-s.diff((nu+nt)*s.diff(k,coords[j]),coords[j]) for j in range(2))-P+e+D
        re=sum(velocity[j]*s.diff(e,coords[j])-s.diff((nu+nt/1.3)*s.diff(e,coords[j]),coords[j]) for j in range(2))-1.44*P*e/(k+1e-20)+1.92*(1-.3*s.exp(-rt**2))*e*e/(k+1e-20)-E
        class Manufactured:
            k_inf=1e-6;et_inf=1e-7
            def __call__(self,xy):
                x,y=xy[:,0:1],xy[:,1:2];q=.2+.03*x+.02*y+.01*x*x;e=.04+.003*x*x+.002*y*y
                return dict(u=1+x*x+.3*x*y+.1*y**3,v=.2*x**3-.4*x*y+.2*y*y,p=.1*x*x+y,q=q,k=q*q,ep_tilde=e)
        xy=torch.tensor([[.2,.3],[-.1,.15]],requires_grad=True);_,d=physics(Manufactured(),xy,nu)
        expressions={'momentum_x':mom[0],'momentum_y':mom[1],'k_residual_raw':rk,'ep_residual_raw':re,'D':D,'E':E,'production':P,'ep':e+D}
        for name,expr in expressions.items():
            expected=np.asarray(s.lambdify((x,y),expr,'numpy')(xy.detach().numpy()[:,0],xy.detach().numpy()[:,1]))
            np.testing.assert_allclose(d[name].detach().numpy().reshape(-1),expected,rtol=1e-10,atol=1e-11,err_msg=name)
    def test_full_backward_has_finite_gradients(self):
        ki,ei=freestream(1/6e6);m=FlowModel(ki,ei,8,2)
        xy=torch.tensor([[.3,.1],[.5,-.1],[-.05,.03]],requires_grad=True)
        r,d=physics(m,xy,1/6e6);loss=sum(z.square().mean() for z in r.values());loss.backward()
        for name in ['flow','pressure','tke','dissipation']:
            grads=[p.grad for p in getattr(m,name).parameters() if p.grad is not None]
            self.assertTrue(grads);self.assertTrue(all(torch.isfinite(z).all() for z in grads))
        torch.testing.assert_close(d['ep'],d['ep_tilde']+d['D'])

if __name__=='__main__':unittest.main()
