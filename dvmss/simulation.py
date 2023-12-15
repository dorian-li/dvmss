import numpy as np

from .agent import MagAgent
from .flight import Flight
from .geomag import GeomagneticReferenceField


class Simulation:
    def __init__(
        self, mag_agent: MagAgent, geomag: GeomagneticReferenceField, flight: Flight
    ) -> None:
        self.mag_agent = mag_agent
        self.geomag = geomag
        self.flight = flight
