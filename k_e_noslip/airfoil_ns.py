# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from sympy import Symbol, Or, Max

import physicsnemo.sym
from physicsnemo.sym.key import Key
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Line, Channel2D
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.geometry import Parameterization, Parameter
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.utils.io import InferencerPlotter

from custom_airfoil_geometry import AirfoilInChannel
from sampling_solver import InferenceScheduledResamplingSolver, sampling_setup


@physicsnemo.sym.main(config_path="conf", config_name="config_ns")
def run(cfg: PhysicsNeMoConfig) -> None:
    ##########
    # Geometry
    ##########

    x, y = Symbol("x"), Symbol("y")  # Physical coordinates

    # A range of min/max values for the airfoil parameters
    # fmt: off
    parameterization = Parameterization({
        Parameter("alpha"): (float(cfg.custom.alpha_min), float(cfg.custom.alpha_max)),
        Parameter("camber"): (float(cfg.custom.camber_min), float(cfg.custom.camber_max)),
        Parameter("thickness"): (float(cfg.custom.thickness_min), float(cfg.custom.thickness_max)),
    })
    # fmt: on

    channel_length = 15.0 / 2
    channel_height = 10.0 / 2
    x_fraction = 0.3

    channel = Channel2D(
        (-channel_length * x_fraction, -channel_height / 2),
        (channel_length * (1 - x_fraction), channel_height / 2),
    )
    inlet = Line(
        (-channel_length * x_fraction, -channel_height / 2),
        (-channel_length * x_fraction, channel_height / 2),
        normal=1,
    )
    outlet = Line(
        (channel_length * (1 - x_fraction), -channel_height / 2),
        (channel_length * (1 - x_fraction), channel_height / 2),
        normal=1,
    )

    near_field_channel_length = 1.6
    near_field_channel_height = 1.0
    x_fraction_near_field = 0.2
    near_xmin = -near_field_channel_length*x_fraction_near_field
    near_xmax = near_field_channel_length*(1-x_fraction_near_field)
    near_ymin = -near_field_channel_height/2
    near_ymax = near_field_channel_height/2
    volume_near_field = AirfoilInChannel(
        x_min=-near_field_channel_length * x_fraction_near_field,
        y_min=-near_field_channel_height / 2,
        x_max=near_field_channel_length * (1 - x_fraction_near_field),
        y_max=near_field_channel_height / 2,
        params=parameterization,
        include_channel_boundary=False,
        airfoil_resolution=int(cfg.custom.airfoil_resolution),
    )

    far_criteria = Or(x <= near_xmin, x >= near_xmax, y <= near_ymin, y >= near_ymax)
    far_area = channel_length*channel_height-near_field_channel_length*near_field_channel_height

    ###########
    # Equations
    ###########
    u_inlet = 1.0
    chord = 1.0
    reynolds_number = float(cfg.custom.reynolds_number)
    if reynolds_number <= 0:
        raise ValueError("Reynolds number must be positive.")
    nu = u_inlet * chord / reynolds_number  # kinematic viscosity

    navier_stokes = NavierStokes(nu=nu, rho=1.0, dim=2, time=False)

    ###########
    # Networks
    ###########
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("alpha"), Key("camber"), Key("thickness")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )

    #############
    # Constraints
    #############

    domain = Domain()
    caches, interval, cached = sampling_setup(cfg)

    nodes: list = navier_stokes.make_nodes() + [flow_net.make_node(name="flow_network")]

    # inlet
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=cached("inlet", inlet, channel_height),
        outvar={"u": u_inlet, "v": 0},
        batch_size=cfg.batch_size.inlet,
        fixed_dataset=False,
        num_workers=0,
        parameterization=parameterization,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=cached("outlet", outlet, channel_height),
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        fixed_dataset=False,
        num_workers=0,
        parameterization=parameterization,
    )
    domain.add_constraint(outlet, "outlet")

    # freestream (channel top-bot)
    top_bot = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=cached("top_bot", channel, 2*channel_length),
        outvar={"u": u_inlet, "v": 0},
        batch_size=cfg.batch_size.top_bot,
        fixed_dataset=False,
        num_workers=0,
        parameterization=parameterization,
    )
    domain.add_constraint(top_bot, "top_bot")

    # no slip
    airfoil = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=cached("airfoil", volume_near_field),
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.airfoil,
        fixed_dataset=False,
        num_workers=0,
        parameterization=parameterization,
    )
    domain.add_constraint(airfoil, "airfoil")

    # interior contraints
    interior_far_field = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=cached("interior_far_field", channel, far_area),
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior_far_field,
        criteria=far_criteria,
        fixed_dataset=False,
        num_workers=0,
        parameterization=parameterization,
    )
    domain.add_constraint(interior_far_field, "interior_far_field")

    interior_near_field = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=cached("interior_near_field", volume_near_field),
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior_near_field,
        lambda_weighting={
            "continuity": 2 * Max(Symbol("sdf"), 0),
            "momentum_x": 2 * Max(Symbol("sdf"), 0),
            "momentum_y": 2 * Max(Symbol("sdf"), 0),
        },
        fixed_dataset=False,
        num_workers=0,
        parameterization=parameterization,
    )
    domain.add_constraint(interior_near_field, "interior_near_field")

    ##############
    # Inferencers
    ##############
    inferencer_id: int = 0

    def add_inferencer(
        alpha_inference: float = 0.0,
        camber_inference: float = 0.0,
        thickness_inference: float = 0.12,
        n_interior_points: int = 20000,
        n_boundary_points: int = 512,
    ) -> None:
        """
        Adds an inferencer to the domain.

        Note: This is not a pure function! Applies nonlocal modifications to:
            - `inferencer_id`, such that inferencer file outputs are unique.
            - `domain`

        Parameters
        ----------
        alpha_inference: float
            The angle of attack of the inference case to add.
        camber_inference: float
            The camber of the airfoil in the inference case to add, as a fraction of chord.
        thickness_inference: float
            The thickness of the airfoil in the inference case to add, as a fraction of chord.
        n_interior_points: int
            The number of interior points to sample from each the near- and far-fields of each inferencer.
        n_boundary_points: int
            The number of boundary points to sample from each inferencer.

        Returns
        -------
        None (makes in-place modifications to `domain` and `inferencer_id`)
        """
        nonlocal inferencer_id

        parameterization = Parameterization(
            {
                Parameter("alpha"): (alpha_inference, alpha_inference),
                Parameter("camber"): (camber_inference, camber_inference),
                Parameter("thickness"): (thickness_inference, thickness_inference),
            }
        )

        sample_near_field = volume_near_field.sample_interior(
            n_interior_points,
            rand_seed=inferencer_id,
            parameterization=parameterization,
        )
        sample_boundary = volume_near_field.sample_boundary(
            n_boundary_points,
            rand_seed=inferencer_id,
            parameterization=parameterization,
        )
        sample_far_field = channel.sample_interior(
            n_interior_points,
            criteria=far_criteria,
            quasirandom=True,
            parameterization=parameterization,
        )
        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar={
                "x": sample_near_field["x"],
                "y": sample_near_field["y"],
                "alpha": alpha_inference * np.ones_like(sample_near_field["y"]),
                "camber": camber_inference * np.ones_like(sample_near_field["y"]),
                "thickness": thickness_inference * np.ones_like(sample_near_field["y"]),
            },
            output_names=["u", "v", "p"],
            batch_size=n_interior_points,
            plotter=InferencerPlotter(),
        )
        domain.add_inferencer(grid_inference, f"near_field_{inferencer_id + 1}")

        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar={
                "x": sample_boundary["x"],
                "y": sample_boundary["y"],
                "alpha": alpha_inference * np.ones_like(sample_boundary["y"]),
                "camber": camber_inference * np.ones_like(sample_boundary["y"]),
                "thickness": thickness_inference * np.ones_like(sample_boundary["y"]),
                # Retain surface quadrature and fluid-outward normals for drag analysis.
                "normal_x": sample_boundary["normal_x"],
                "normal_y": sample_boundary["normal_y"],
                "area": sample_boundary["area"],
            },
            output_names=["u", "v", "p", "u__x", "u__y", "v__x", "v__y"],
            batch_size=n_boundary_points,
            requires_grad=True,  # Record wall velocity derivatives using AD.
            plotter=InferencerPlotter(),
        )
        domain.add_inferencer(grid_inference, f"boundary_{inferencer_id + 1}")

        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar={
                "x": sample_far_field["x"],
                "y": sample_far_field["y"],
                "alpha": alpha_inference * np.ones_like(sample_far_field["y"]),
                "camber": camber_inference * np.ones_like(sample_far_field["y"]),
                "thickness": thickness_inference * np.ones_like(sample_far_field["y"]),
            },
            output_names=["u", "v", "p"],
            batch_size=n_interior_points,
            plotter=InferencerPlotter(),
        )
        domain.add_inferencer(grid_inference, f"far_field_{inferencer_id + 1}")

        inferencer_id += 1

    add_inferencer(alpha_inference=(float(cfg.custom.alpha_min)+float(cfg.custom.alpha_max))/2,
                   camber_inference=(float(cfg.custom.camber_min)+float(cfg.custom.camber_max))/2,
                   thickness_inference=(float(cfg.custom.thickness_min)+float(cfg.custom.thickness_max))/2)

    # Random inferencers (disabled for the single 0-degree baseline).
    # rng = np.random.default_rng(42)

    # for i in range(6):
    #     alpha_inference = rng.uniform(float(cfg.custom.alpha_min), float(cfg.custom.alpha_max))
    #     camber_inference = rng.uniform(float(cfg.custom.camber_min), float(cfg.custom.camber_max))
    #     thickness_inference = rng.uniform(float(cfg.custom.thickness_min), float(cfg.custom.thickness_max))
    #     add_inferencer(
    #         alpha_inference=alpha_inference,
    #         camber_inference=camber_inference,
    #         thickness_inference=thickness_inference,
    #     )

    ##############
    # Solver
    ##############
    

    slv = InferenceScheduledResamplingSolver(cfg, domain, caches, interval)
    slv.solve()

    # Record the results explicitly
    
    print("Triggering manual evaluation for ParaView...")
    slv.eval()

if __name__ == "__main__":
    run()
