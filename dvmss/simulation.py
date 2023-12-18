from dataclasses import astuple
from functools import reduce
from typing import List, Union

import numpy as np
from metalpy.scab.builder.simulation_builder import SimulationBuilder
from metalpy.scab.formatted.formatted import Formatted
from metalpy.scab.potential_fields.magnetics.simulation import Simulation3DDipoles
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

    def perm_interf(self, detectors: DetectorCollection):
        builder_d = SimulationBuilder.of(Simulation3DDipoles)
        sources = tuple(
            list(astuple(s.location)) for s in self.mag_agent.config.interf.perm.sources
        )
        builder_d.sources(*sources)
        rx_loc = np.array([astuple(d.location) for d in detectors.detectors])
        
        builder_d.receivers(rx_loc, ["bx", "by", "bz"])
        builder_d.patched(Formatted())
        model = [s.moment_vector for s in self.mag_agent.config.interf.perm.sources]
        d = builder_d.build().dpred(model)
        print(d)

    def sample(self, detectors: DetectorCollection):
        return detectors
