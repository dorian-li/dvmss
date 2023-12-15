from typing import List, Union

import numpy as np

from .agent import MagAgent
from .detector import Detector
from .flight import Flight
from .geomag import GeomagneticReferenceField


class Simulation:
    def __init__(
        self, mag_agent: MagAgent, geomag: GeomagneticReferenceField, flight: Flight
    ) -> None:
        self.mag_agent = mag_agent
        self.geomag = geomag
        self.flight = flight

    def sample(self, detectors: Union[Detector, List[Detector]]):
        return detectors
