import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model import physics

class WallLimitTests(unittest.TestCase):
    def test_modified_epsilon_zero_physical_epsilon_finite(self):
        class WallExpansion:
            k_inf=1e-6;et_inf=1e-7
            def __call__(self,xy):
                x,y=xy[:,0:1],xy[:,1:2]
                z=0*x**3+0*y**3 # keep zero analytic derivatives on the autograd graph
                q=2*y+z
                return dict(u=3*y+z,v=z,p=z,q=q,k=q*q,ep_tilde=4*y*y+z)
        xy=torch.tensor([[0.,1e-5],[0.,1e-6],[0.,1e-7]],dtype=torch.float64,requires_grad=True)
        _,d=physics(WallExpansion(),xy,nu=.01)
        self.assertAlmostEqual(float(d['ep'][-1].detach()),.08,places=11)
        self.assertLess(float(d['ep_tilde'][-1].detach()),1e-12)
        self.assertTrue(torch.isfinite(d['ep_equation']).all())

if __name__=='__main__':unittest.main()
