from dataclasses import astuple
from functools import reduce
from typing import List, Union

import numpy as np
from metalpy.mepa.process_executor import ProcessExecutor
from metalpy.scab.builder.simulation_builder import SimulationBuilder
from metalpy.scab.formatted.formatted import Formatted
from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Obj2
from metalpy.scab.modelling.transform import Rotation
from metalpy.scab.potential_fields.magnetics.simulation import Simulation3DDipoles
from metalpy.utils.bounds import Bounds
from SimPEG.potential_fields.magnetics.simulation import Simulation3DIntegral

from .agent import MagAgent
from .detector import DetectorCollection
from .flight import Flight, VehicleState
from .geomag import GeomagData, GeomagRefField


class Simulation:
    def __init__(
        self, mag_agent: MagAgent, geomag: GeomagRefField, flight: Flight
    ) -> None:
        self.mag_agent = mag_agent
        self.geomag = geomag
        self.flight = flight
        self._cached_background_field: GeomagData = None

    # def voxelize_mag_agent(self):
    #     vehicle = self.mag_agent.config.vehicle
    #     model_3d_bounds = Bounds(vehicle.model_3d.bounds).extent
    #     if vehicle.wingspan > vehicle.length:
    #         wing_axis = model_3d_bounds.argmax()
    #     scale_factor = (
    #         self.mag_agent.config.vehicle.real_longest_axis_length
    #         / Bounds(vehicle_3d.bounds).extent.max()
    #     )
    #     agent_obj = Obj2(
    #         model=vehicle.model_3d,
    #         scale=scale_factor,
    #         surface_range=[-0.1, 0.1],
    #         subdivide=True,
    #         ignore_surface_check=True,
    #     )
    #     to_scene_center = -1 * agent_obj.center
    #     # Rotation(180, 0, 0, degrees=True, seq="zyx") rotate to Rotation(0, 0, 0, degrees=True, seq="zyx")
    #     to_northward =
    #     agent_obj.translate(*to_scene_center).apply()
    #     scene = Scene.of()

    @property
    def background_field(self):
        if self._cached_background_field is None:
            return self.geomag.query(
                self.flight.date,
                self.flight.query(VehicleState.LATITUDE),
                self.flight.query(VehicleState.LONGITUDE),
                self.flight.query(VehicleState.ELEVATION),
            )
        return self._cached_background_field

    def compute_perm_interf_vector(self, detectors: DetectorCollection):
        builder_d = SimulationBuilder.of(Simulation3DDipoles)
        sources = tuple(
            list(astuple(s.location)) for s in self.mag_agent.config.interf.perm.sources
        )
        builder_d.sources(*sources)
        rx_loc = np.array([astuple(d.location) for d in detectors.detectors])

        builder_d.receivers(rx_loc, ["bx", "by", "bz"])
        builder_d.patched(Formatted())
        model = np.array(
            [s.moment_vector for s in self.mag_agent.config.interf.perm.sources]
        ).flatten("F")
        d = builder_d.build().dpred(model)
        return d

    def compute_induced_interf_vector(self, detectors: DetectorCollection):
        builder = SimulationBuilder.of(Simulation3DIntegral)
        builder.source_field(strength=1, inc=1, dec=1)
        rx_loc = np.array([astuple(d.location) for d in detectors.detectors])
        builder.receivers(rx_loc, ["bx", "by", "bz"])
        builder.vector_model()
        builder.active_mesh(self.model_mesh)
        builder.store_sensitivities(True)
        induced_simulation = builder.build()
        self._sensitivity = induced_simulation.G

    def sample(self, detectors: DetectorCollection):
        return detectors
