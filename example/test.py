import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

import numpy as np
from numpy.testing import assert_allclose

from dvmss.geomag import IGRF, GeomagElem

geo_igrf = IGRF.query(
    date=datetime(2022, 1, 1),
    longitude=-113.64250,
    latitude=60.10861,
    elevation=0,
)
orient = geo_igrf.get_orientations()
orient_2 = (
    geo_igrf.query(GeomagElem.NORTH, GeomagElem.EAST, GeomagElem.VERTICAL).to_numpy()
    / geo_igrf.query(GeomagElem.TOTAL).to_numpy()
)
print(orient)  # ENU
print(orient_2)  # NED
