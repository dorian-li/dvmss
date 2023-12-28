import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

import numpy as np
from geoana import spatial, utils
from geoana.em import static
from numpy.testing import assert_allclose
from SimPEG.utils.mat_utils import dip_azimuth2cartesian

from dvmss.geomag import IGRF, GeomagElem

wing_0_loc = np.array([[-5.5, 1.1, 0.32]])
wing_1_loc = np.array([[5.5, 1.1, 0.32]])
tail_loc = np.array([[0, -8, -0.7]])

geo_orient = dip_azimuth2cartesian(20.242, -2.186)
geo_orient = geo_orient.T

for moment_0 in np.linspace(0.1, 200, 2000):
    for moment_1 in np.linspace(0.1, 200, 2000):
        dipole_0 = static.MagneticDipoleWholeSpace(
            location=np.array([0.0, 1.9, 0.0]),
            orientation=dip_azimuth2cartesian(10, 90),
            moment=moment_0,
        )

        dipole_1 = static.MagneticDipoleWholeSpace(
            location=np.array([0.0, 2.7, 0.0]),
            orientation=dip_azimuth2cartesian(-10, 270),
            moment=moment_1,
        )
        wing_0_source0 = dipole_0.magnetic_flux_density(wing_0_loc)
        wing_0_source1 = dipole_1.magnetic_flux_density(wing_0_loc)
        wing_0 = wing_0_source0 + wing_0_source1
        wing_0 *= 1e9
        wing_0_tmi = float(np.abs(np.dot(wing_0, geo_orient)[0][0]))

        # wing_1
        wing_1_source0 = dipole_0.magnetic_flux_density(wing_1_loc)
        wing_1_source1 = dipole_1.magnetic_flux_density(wing_1_loc)
        wing_1 = wing_1_source0 + wing_1_source1
        wing_1 *= 1e9
        wing_1_tmi = float(np.abs(np.dot(wing_1, geo_orient)[0][0]))

        # tail
        tail_source0 = dipole_0.magnetic_flux_density(tail_loc)
        tail_source1 = dipole_1.magnetic_flux_density(tail_loc)
        tail = tail_source0 + tail_source1
        tail *= 1e9
        tail_tmi = float(np.abs(np.dot(tail, geo_orient)[0][0]))

        if (
            (wing_0_tmi > 13 and wing_0_tmi < 18)
            and (wing_1_tmi > 13 and wing_1_tmi < 18)
            and (tail_tmi > 1.2 and tail_tmi < 1.8)
        ):
            print(f"{moment_0=}, {moment_1=}")
            print(f"{wing_0_tmi=}, {wing_1_tmi=}, {tail_tmi=}")
