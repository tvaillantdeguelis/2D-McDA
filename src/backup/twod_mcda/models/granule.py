from dataclasses import dataclass
import numpy as np

@dataclass
class Granule:
    latitude: np.ndarray
    longitude: np.ndarray
    altitude: np.ndarray

    atb532: np.ndarray
    atb1064: np.ndarray

    surface_mask: np.ndarray | None = None
    cloud_mask: np.ndarray | None = None
    aerosol_mask: np.ndarray | None = None
    