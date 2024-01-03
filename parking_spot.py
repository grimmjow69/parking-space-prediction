import numpy as np


class ParkingSpot:
    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates

    def get_polygon(self):
        return np.array(self.coordinates, np.int32).reshape((-1, 1, 2))
