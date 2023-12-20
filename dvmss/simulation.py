from dataclasses import astuple
from functools import reduce
from typing import List, Union

import numpy as np
from metalpy.mepa.process_executor import ProcessExecutor
from metalpy.scab import Tied
from metalpy.scab.builder.simulation_builder import SimulationBuilder
from metalpy.scab.formatted import Formatted
from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Obj2
from metalpy.scab.modelling.transform import Rotation
from metalpy.scab.potential_fields.magnetics.simulation import Simulation3DDipoles
from metalpy.scab.utils.format import format_pandas
from metalpy.utils.bounds import Bounds
from SimPEG.potential_fields.magnetics.simulation import Simulation3DIntegral

from .agent import MagAgent
from .detector import DetectorCollection, MagSensor
from .flight import Flight, VehicleState
from .geomag import GeomagData, GeomagElem, GeomagRefField


class Simulation:
    def __init__(
        self, mag_agent: MagAgent, geomag: GeomagRefField, flight: Flight
    ) -> None:
        self.mag_agent = mag_agent
        self.geomag = geomag
        self.flight = flight
        self._cached_background_field: GeomagData = None
        self.mag_agent_mesh = self.voxelize_mag_agent()
        # self.mag_agent_mesh.to_polydata().plot(show_grid=True, show_edges=True, opacity=0.5)

    def voxelize_mag_agent(self):
        scene = Scene.of(
            Obj2(
                model=self.mag_agent.config.vehicle.model_3d,
                surface_range=[-0.1, 0.1],
                subdivide=True,
                ignore_surface_check=True,
            ),
            models=self.mag_agent.config.interf.induced.susceptibility,
        )
        return scene.build(
            cell_size=self.mag_agent.config.interf.induced.voxel_cell_size,
            cache=True,
            executor=ProcessExecutor(),
        )

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

    def compute_geomagnetic_field(self, detectors: DetectorCollection):
        for detector in detectors:
            detector.assign_sensor_data(
                MagSensor.GEO_X,
                self.background_field.query(GeomagElem.EAST),
            )
            detector.assign_sensor_data(
                MagSensor.GEO_Y,
                self.background_field.query(GeomagElem.NORTH),
            )
            detector.assign_sensor_data(
                MagSensor.GEO_Z,
                -1 * self.background_field.query(GeomagElem.VERTICAL),
            )  # 地磁场坐标系向下为正，传感器坐标系向上为正
        return detectors

    def compute_permanent_interf(self, detectors: DetectorCollection):
        builder_d = SimulationBuilder.of(Simulation3DDipoles)
        sources = tuple(
            list(astuple(s.location)) for s in self.mag_agent.config.interf.perm.sources
        )
        builder_d.sources(*sources)
        rx_loc = np.array([astuple(d.location) for d in detectors.items])

        builder_d.receivers(rx_loc, ["bx", "by", "bz"])
        # builder_d.patched(Formatted())
        model = np.array(
            [s.moment_vector for s in self.mag_agent.config.interf.perm.sources]
        ).flatten("F")
        perm_vectors = builder_d.build().dpred(model)  # shape(detector_num * 3,)
        perm_vectors_expanded = np.repeat(
            perm_vectors[:, np.newaxis], len(self.flight._states), axis=1
        ).T  # (flight_len, detector_num * 3)
        for i, detector in enumerate(detectors):
            detector.assign_sensor_data(MagSensor.PREM_X, perm_vectors_expanded[:, 3 * i])
            detector.assign_sensor_data(MagSensor.PREM_Y, perm_vectors_expanded[:, 3 * i + 1])
            detector.assign_sensor_data(MagSensor.PREM_Z, perm_vectors_expanded[:, 3 * i + 2])
        return detectors

    def compute_induced_interf(self, detectors: DetectorCollection):
        components = ["bx", "by", "bz"]
        builder = SimulationBuilder.of(Simulation3DIntegral)
        builder.source_field(strength=1, inc=1, dec=1)
        rx_loc = np.array([astuple(d.location) for d in detectors.items])
        builder.receivers(rx_loc, components)
        builder.vector_model()
        builder.active_mesh(self.mag_agent_mesh)
        builder.store_sensitivities(True)
        induced_simulation = builder.build()

        G = induced_simulation.G
        print(f"{G.shape=}")

        bg_field_orienttation = (
            self.background_field.get_orientations()
        )  # shape: (flight_len, 3)
        model_mag_direct = self.flight.att_rot.apply(
            bg_field_orienttation, inverse=True
        )  # shape: (flight_len, 3)
        model_scalar = (
            self.mag_agent_mesh.get_active_model()
        )  # shape: (active_mesh_num, )
        model_vector = np.einsum("i, jk -> jik", model_scalar, model_mag_direct)
        model_vector = model_vector.reshape(
            (model_mag_direct.shape[0], model_scalar.shape[0] * 3),
            order="F",
        ).T  # shape: (active_mesh_num * 3, flight_len)
        result = G @ model_vector  # shape: (rx_num * 3, flight_len)
        # ret = format_pandas(result, components, rx_loc)
        print(result.shape)

    def sample(self, detectors: DetectorCollection):
        detectors = self.compute_geomagnetic_field(detectors)
        detectors = self.compute_permanent_interf(detectors)
        detectors = self.compute_induced_interf(detectors)
        detectors = self.merge_sensor_component(detectors)

        return detectors
