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
from .utils import NED_to_ENU, project_vectors_to_orientations


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
            models=self.mag_agent.config.interference.induced_field.susceptibility,
        )
        return scene.build(
            cell_size=self.mag_agent.config.interference.induced_field.voxel_cell_size,
            cache=True,
            executor=ProcessExecutor(),
        )

    @property
    def background_field(self):
        if self._cached_background_field is None:
            self._cached_background_field = self.geomag.query(
                self.flight.date,
                self.flight.query(VehicleState.LATITUDE),
                self.flight.query(VehicleState.LONGITUDE),
                self.flight.query(VehicleState.ELEVATION),
            )  # NED坐标系
        return self._cached_background_field

    @property
    def tmi_projections(self):
        return self.flight.att_rot.apply(
            self.background_field.get_orientations(), inverse=True
        )  # (flight_len, 3), ENU坐标系

    def compute_background_field_vector(self, detectors: DetectorCollection):
        bg_NED = self.background_field.query(
            GeomagElem.NORTH, GeomagElem.EAST, GeomagElem.VERTICAL
        )  # NED坐标系
        bg_body_frame = self.flight.att_rot.apply(
            NED_to_ENU(bg_NED.to_numpy()),
            inverse=True,
        )  # shape: (flight_len, 3)
        for detector in detectors:
            # to ENU坐标系
            detector.assign_sensor_data(MagSensor.GEO_X, bg_body_frame[:, 0])
            detector.assign_sensor_data(MagSensor.GEO_Y, bg_body_frame[:, 1])
            detector.assign_sensor_data(MagSensor.GEO_Z, bg_body_frame[:, 2])
        return detectors

    def compute_permanent_interf_vector(self, detectors: DetectorCollection):
        builder = SimulationBuilder.of(Simulation3DDipoles)
        source_locs = tuple(
            list(astuple(s.location)) for s in self.mag_agent.config.interference.permanent_field.sources
        )
        builder.sources(*source_locs)
        detector_locs = np.array([astuple(d.location) for d in detectors.items])

        builder.receivers(detector_locs, ["bx", "by", "bz"])
        model = np.array(
            [s.moment_vector.to_numpy() for s in self.mag_agent.config.interference.permanent_field.sources]
        ).flatten(
            "F"
        )  # (source_num * 3,)
        perm_vectors = builder.build().dpred(model)  # (detector_num * 3,)
        perm_vectors_expanded = np.repeat(
            perm_vectors[:, np.newaxis], len(self.flight._states), axis=1
        ).T  # (flight_len, detector_num * 3)
        for i, detector in enumerate(detectors):
            detector.assign_sensor_data(
                MagSensor.PERM_X, perm_vectors_expanded[:, 3 * i]
            )
            detector.assign_sensor_data(
                MagSensor.PERM_Y, perm_vectors_expanded[:, 3 * i + 1]
            )
            detector.assign_sensor_data(
                MagSensor.PERM_Z, perm_vectors_expanded[:, 3 * i + 2]
            )
        return detectors

    def compute_induced_interf_vector(self, detectors: DetectorCollection):
        components = ["bx", "by", "bz"]
        builder = SimulationBuilder.of(Simulation3DIntegral)
        builder.source_field(strength=1, inc=1, dec=1)
        dectector_locs = np.array([astuple(d.location) for d in detectors.items])
        builder.receivers(dectector_locs, components)
        builder.vector_model()
        builder.active_mesh(self.mag_agent_mesh)
        builder.store_sensitivities(True)

        kernel = builder.build().G  # (detector_num * 3, active_mesh_num * 3)
        model_mag_direct = self.tmi_projections  # (flight_len, 3)
        model_scalar = self.mag_agent_mesh.get_active_model()  # (active_mesh_num, )
        # TODO: 因为每一个体素网格的磁化方向都是一样的，可以只计算一个来优化👇
        model_vector = np.einsum(
            "i, jk -> jik", model_scalar, model_mag_direct
        )  # (flight_len, active_mesh_num, 3)
        model_vector = model_vector.reshape(
            (model_mag_direct.shape[0], model_scalar.shape[0] * 3),
            order="F",
        ).T  # (active_mesh_num * 3, flight_len)
        induced_vectors = kernel @ model_vector  # (detector_num * 3, flight_len)
        induced_vectors = induced_vectors.T  # (flight_len, detector_num * 3)

        for i, detector in enumerate(detectors):
            detector.assign_sensor_data(MagSensor.INDUCED_X, induced_vectors[:, 3 * i])
            detector.assign_sensor_data(
                MagSensor.INDUCED_Y, induced_vectors[:, 3 * i + 1]
            )
            detector.assign_sensor_data(
                MagSensor.INDUCED_Z, induced_vectors[:, 3 * i + 2]
            )
        return detectors

    def merge_sensor_vector_component(self, detectors: DetectorCollection):
        for detector in detectors:
            detector.assign_sensor_data(
                MagSensor.B_X,
                detector.sensor_data[MagSensor.GEO_X]
                + detector.sensor_data[MagSensor.PERM_X]
                + detector.sensor_data[MagSensor.INDUCED_X],
            )
            detector.assign_sensor_data(
                MagSensor.B_Y,
                detector.sensor_data[MagSensor.GEO_Y]
                + detector.sensor_data[MagSensor.PERM_Y]
                + detector.sensor_data[MagSensor.INDUCED_Y],
            )
            detector.assign_sensor_data(
                MagSensor.B_Z,
                detector.sensor_data[MagSensor.GEO_Z]
                + detector.sensor_data[MagSensor.PERM_Z]
                + detector.sensor_data[MagSensor.INDUCED_Z],
            )
        return detectors

    def compute_tmi(self, detectors: DetectorCollection):
        for detector in detectors:
            detector.assign_sensor_data(
                MagSensor.TMI,
                project_vectors_to_orientations(
                    detector.sensor_data[[MagSensor.B_X, MagSensor.B_Y, MagSensor.B_Z]],
                    self.tmi_projections,
                ),
            )
            detector.assign_sensor_data(
                MagSensor.PERM_TMI,
                project_vectors_to_orientations(
                    detector.sensor_data[
                        [MagSensor.PERM_X, MagSensor.PERM_Y, MagSensor.PERM_Z]
                    ],
                    self.tmi_projections,
                ),
            )
            detector.assign_sensor_data(
                MagSensor.INDUCED_TMI,
                project_vectors_to_orientations(
                    detector.sensor_data[
                        [MagSensor.INDUCED_X, MagSensor.INDUCED_Y, MagSensor.INDUCED_Z]
                    ],
                    self.tmi_projections,
                ),
            )
            detector.assign_sensor_data(
                MagSensor.GEO_T,
                np.linalg.norm(
                    detector.sensor_data[
                        [MagSensor.GEO_X, MagSensor.GEO_Y, MagSensor.GEO_Z]
                    ],
                    axis=1,
                ),
            )
        return detectors

    def sample(self, detectors: DetectorCollection):
        detectors = self.compute_background_field_vector(detectors)
        detectors = self.compute_permanent_interf_vector(detectors)
        detectors = self.compute_induced_interf_vector(detectors)
        detectors = self.merge_sensor_vector_component(detectors)
        detectors = self.compute_tmi(detectors)

        return detectors
