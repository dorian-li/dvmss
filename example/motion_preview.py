from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sgl2020 import SGL2020

from dvmss import FlightMotion, MagDipoleParam, Simulator, agents
from dvmss.utils import pred_data_to_df

surv = SGL2020()
flt_d = surv.source(
    ["lon", "lat", "utm_z", "ins_pitch", "ins_roll", "ins_yaw", "mag_1_c"]
).take(include_line=True)

agent = agents.cessna_172()
# flux_pos = np.array([[-3.19112678, -1.1070188, -0.91339862]])
op_pos = np.array([[-0.19112678, -1.1070188, -0.91339862]])
flux_pos = op_pos
perm_param = [
    MagDipoleParam(
        moment=1e6,
        orientation=(7.69, -75.13),
        position=(4.30506149e-04, 3.39457732e00, -1.03428663e-02),
    ),
]
susceptibility = 0.3
cell_size = 1
size_unit = "scale"

sim = Simulator()
sim.agent(agent)
sim.vector_receiver(flux_pos)
sim.scalar_receiver(op_pos)
sim.permanent(perm_param)
sim.induced(susceptibility, cell_size, size_unit)

sim_ds = pd.DataFrame()
for line, line_d in flt_d.groupby("line"):
    if line == 1006.06:
        sim.motion(
            FlightMotion(
                datetime(2023, 8, 26),
                line_d["lon"],
                line_d["lat"],
                line_d["utm_z"] / 1000,
                line_d["ins_pitch"],
                line_d["ins_roll"],
                -line_d["ins_yaw"],
            )
        )
        sim.background_field(line_d["mag_1_c"])
        sim.motion_preview("motion_preview.mp4")
