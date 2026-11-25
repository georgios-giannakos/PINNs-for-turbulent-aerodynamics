# SPDX-License-Identifier: Apache-2.0
"""Integrate signed pressure and modeled shear on physical wall weights."""
import argparse
import json
from pathlib import Path
import numpy as np
from vtk import vtkXMLPolyDataReader
from vtk.util.numpy_support import vtk_to_numpy


def read(path):
    r=vtkXMLPolyDataReader()
    r.SetFileName(str(path)); r.Update()
    o=r.GetOutput()
    if not o.GetNumberOfPoints():
        raise ValueError(f"No points: {path}")
    pd=o.GetPointData()
    a={pd.GetArrayName(i):vtk_to_numpy(pd.GetArray(i)).reshape(-1)
       for i in range(pd.GetNumberOfArrays()) if pd.GetArray(i).GetNumberOfComponents()==1}
    a["point_x"]=vtk_to_numpy(o.GetPoints().GetData())[:,0]
    return a


def main():
    p=argparse.ArgumentParser()
    p.add_argument("inferencers",type=Path)
    p.add_argument("--output",type=Path,default=Path("surface_coefficients.csv"))
    p.add_argument("--p-inf",type=float)
    args=p.parse_args()
    b=read(args.inferencers/"boundary_1.vtp")
    f=read(args.inferencers/"far_field_1.vtp")
    keys=("x_wall","y_wall","p","wall_tau_model","normal_x","normal_y","tangent_x",
          "tangent_y","area","y_plus_wall","normal_distance","u_tau")
    for key in keys:
        if key not in b or not np.isfinite(b[key]).all():
            raise ValueError(f"Missing/nonfinite boundary field: {key}")
    upstream=f["point_x"]<-1.5
    if args.p_inf is None and (not upstream.any() or not np.isfinite(f["p"][upstream]).all()):
        raise ValueError("No finite upstream pressure; specify --p-inf.")
    pinf=float(f["p"][upstream].mean()) if args.p_inf is None else args.p_inf
    w=b["area"].astype(np.float64)
    if np.any(w<=0):
        raise ValueError("Expected positive physical quadrature.")
    cp=2*(b["p"]-pinf); cf=2*b["wall_tau_model"]
    cdp=float(np.sum(w*cp*b["normal_x"])); cdf=float(np.sum(w*cf*b["tangent_x"]))
    clp=float(np.sum(w*cp*b["normal_y"])); clf=float(np.sum(w*cf*b["tangent_y"]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(args.output,np.column_stack([b["x_wall"],b["y_wall"],cp,cf,b["y_plus_wall"],
                 b["normal_distance"],b["u_tau"],w]),delimiter=",",comments="",
                 header="x_wall,y_wall,Cp_at_P,Cf_model,y_plus,normal_distance,u_tau,area")
    print(json.dumps(dict(p_inf=pinf,CD_pressure=cdp,CD_friction=cdf,CD=cdp+cdf,
        CL_pressure=clp,CL_friction=clf,CL=clp+clf,perimeter=float(w.sum()),
        normal_closure_x=float(np.sum(w*b["normal_x"])),
        normal_closure_y=float(np.sum(w*b["normal_y"]))),indent=2))


if __name__=="__main__":
    main()
