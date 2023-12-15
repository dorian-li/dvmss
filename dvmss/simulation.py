from typing import List, Union

import numpy as np

from .agent import MagAgent
from .detector import DetectorGroup
from .flight import Flight
from .geomag import GeomagneticReferenceField


class Simulation:
    def __init__(
        self, mag_agent: MagAgent, geomag: GeomagneticReferenceField, flight: Flight
    ) -> None:
        self.mag_agent = mag_agent
        self.geomag = geomag
        self.flight = flight
    
    def get_geomag(self):
        return self.geomag.get_magnetic_field(self.flight.date, )

    def sample(self, detectors: DetectorGroup):

        return detectors
