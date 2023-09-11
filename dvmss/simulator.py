from collections.abc import Iterable
from functools import cached_property

import numpy as np
import pandas as pd
import pyvista as pv
from discretize.utils import mkvc
from geoana.em.static import MagneticDipoleWholeSpace
from loguru import logger
from metalpy.mepa.process_executor import ProcessExecutor
from metalpy.scab.builder.simulation_builder import SimulationBuilder
from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Obj2
from SimPEG.potential_fields.magnetics.simulation import Simulation3DIntegral
from SimPEG.utils.mat_utils import dip_azimuth2cartesian
from tqdm import tqdm

from .flight_motion import FlightMotion
from .mag_dipole_param import MagDipoleParam
from .utils import GeomagneticField


class Simulator:
    def __init__(self) -> None:
        self._scene = Scene()

    def agent(self, agent):
        self._cell_unit = agent.scale
        _agent = Obj2(
            model=agent.model,
            scale=agent.scale,
            surface_range=[-0.1, 0.1],
            subdivide=True,
            ignore_surface_check=True,
        )
        to_scene_center = -1 * _agent.center
        self._agent = _agent.translate(*to_scene_center).apply(agent.northward)
        logger.debug(f"{self._agent=}")

    def vector_receiver(self, flux_pos):
        self._vector_receiver = flux_pos
        logger.debug(f"{self._vector_receiver=}")

    def scalar_receiver(self, op_pos):
        self._scalar_receiver = op_pos
        logger.debug(f"{self._scalar_receiver=}")

    @property
    def _obs(self):
        return np.r_[self._vector_receiver, self._scalar_receiver]

    def motion(self, motion: FlightMotion):
        if "_geo" in self.__dict__:
            del self.__dict__["_geo"]
        self._motion = motion

    def background_field(self, geo_total):
        if "_geo" in self.__dict__:
            del self.__dict__["_geo"]
        self._background_field = geo_total
        logger.debug(f"{self._background_field=}")

    @cached_property
    def _geo(self):
        return GeomagneticField.make(
            self._motion.lon,
            self._motion.lat,
            self._motion.altitude,
            self._motion.date,
            self._background_field,
        )

    def permanent(self, params: MagDipoleParam | Iterable[MagDipoleParam]):
        if isinstance(params, MagDipoleParam):
            params = [params]
        self._mag_dipoles = [
            MagneticDipoleWholeSpace(
                location=param.position,
                orientation=dip_azimuth2cartesian(*param.orientation),
                moment=param.moment,
            )
            for param in params
        ]
        logger.debug(f"{self._mag_dipoles=}")

    def induced(self, susceptibility, cell_size, cell_unit="scale"):
        self._cell_size = cell_size
        self._cell_unit = self._cell_unit if cell_unit == "scale" else 1
        self._susceptibility = susceptibility

        components = ["bx", "by", "bz"]

        builder = SimulationBuilder.of(Simulation3DIntegral)
        builder.source_field(strength=1, inc=1, dec=1)
        builder.receivers(self._obs, components)
        builder.model_type("vector")
        builder.active_mesh(self.model_mesh)
        builder.store_sensitivities(True)
        induced_simulation = builder.build()
        self._sensitivity = induced_simulation.G

    def _induced_field(self, model):
        return self._sensitivity @ model

    @cached_property
    def model_mesh(self):
        self._scene.append(self._agent, models=self._susceptibility)
        model_mesh = self._scene.build(
            cell_size=self._cell_size * self._cell_unit,
            cache=True,
            executor=ProcessExecutor(),
        )
        logger.debug(f"model mesh size: {model_mesh.get_active_model().shape[0]}")
        return model_mesh

    def scene_plot(self):
        pass

    def motion_preview(self, mov_file):
        mesh = self.model_mesh.to_polydata().threshold(1e-5).copy()
        p = pv.Plotter(window_size=[1600, 912])
        p.open_movie(mov_file, framerate=30, quality=10)
        p.show_axes()
        p.add_mesh(mesh, color="darkturquoise")
        p.add_mesh(mesh.outline_corners())
        p.camera.azimuth = -180
        p.show_grid()
        p.add_arrows(np.array([0, 0, 3]), np.array([0, 1, 0]), color="lightcoral")
        for pitch, roll, yaw in tqdm(self._motion.iteratts()):
            mesh.rotate_x(pitch, inplace=True)
            mesh.rotate_y(roll, inplace=True)
            mesh.rotate_z(yaw, inplace=True)
            p.write_frame()
            mesh.rotate_z(-yaw, inplace=True)
            mesh.rotate_y(-roll, inplace=True)
            mesh.rotate_x(-pitch, inplace=True)

        p.close()

    def dpred(self) -> pd.DataFrame:
        perm_collected = None
        for source in self._mag_dipoles:
            perm_v = source.magnetic_field(self._obs)
            if perm_collected is None:
                perm_collected = perm_v
            else:
                perm_collected = perm_collected + perm_v

        induceds = None
        model_t = self.model_mesh.get_active_model()
        geo_vs = None
        for geo, attitude in tqdm(self.iterframe(), total=self.frame_num):
            model_orientation = attitude.apply(geo.orientation, inverse=True)
            geo_v = geo.total * geo.orientation
            if geo_vs is None:
                geo_vs = geo_v
            else:
                geo_vs = np.r_[geo_vs, geo_v]

            model_v = np.outer(model_t, model_orientation)
            model = mkvc(model_v)
            induced = self._induced_field(model)
            induced *= geo.total
            induced = np.reshape(induced, (-1, 3))
            if induceds is None:
                induceds = induced
            else:
                induceds = np.r_[induceds, induced]

        induceds = np.reshape(induceds, (-1, 2, 3))

        flux_perm_v = perm_collected[0, :]  # shape: (3,)
        op_perm_v = perm_collected[1, :]  # shape: (3,)
        flux_perm_v = np.repeat(
            flux_perm_v[np.newaxis, :], self.frame_num, axis=0
        )  # shape: (frame_num, 3)
        op_perm_v = np.repeat(
            op_perm_v[np.newaxis, :], self.frame_num, axis=0
        )  # shape: (frame_num, 3)

        flux_induced_v = induceds[:, 0, :]  # shape: (frame_num, 3)
        op_induced_v = induceds[:, 1, :]  # shape: (frame_num, 3)

        flux_interf_v = flux_perm_v + flux_induced_v
        op_interf_v = op_perm_v + op_induced_v

        flux_v = flux_interf_v + geo_vs
        flux_v_obs = self._motion.attitude_rotation.apply(flux_v, inverse=True)
        flux_perm_v_obs = self._motion.attitude_rotation.apply(
            flux_perm_v, inverse=True
        )
        flux_induced_v_obs = self._motion.attitude_rotation.apply(
            flux_induced_v, inverse=True
        )
        flux_interf_v_obs = self._motion.attitude_rotation.apply(
            flux_interf_v, inverse=True
        )

        op_v = op_interf_v + geo_vs
        op_s_obs = np.linalg.norm(op_v, axis=1)
        op_perm_s_obs = np.linalg.norm(op_perm_v, axis=1)
        op_induced_s_obs = np.linalg.norm(op_induced_v, axis=1)
        op_interf_s_obs = np.linalg.norm(op_interf_v, axis=1)

        flux_v_gt = self._motion.attitude_rotation.apply(geo_vs, inverse=True)
        op_s_gt = np.linalg.norm(flux_v_gt, axis=1)

        return {
            "vector_receiver": {
                "sensor_data": flux_v_obs,
                "perm": flux_perm_v_obs,
                "induced": flux_induced_v_obs,
                "interf": flux_interf_v_obs,
                "ground_true": flux_v_gt,
            },
            "scalar_receiver": {
                "sensor_data": op_s_obs,
                "perm": op_perm_s_obs,
                "induced": op_induced_s_obs,
                "interf": op_interf_s_obs,
                "ground_true": op_s_gt,
            },
        }

    def assemble_dpred(self):
        pass

    def iterframe(self):
        for geo, attitude in zip(self._geo, self._motion.attitude_rotation):
            yield geo, attitude

    @property
    def frame_num(self):
        return len(self._motion)
