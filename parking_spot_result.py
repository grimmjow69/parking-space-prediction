class ParkingSpotResult:
    def __init__(self, name, occupied):
        self.name = name
        self.occupied = occupied

    def __json__(self):
        return {'name': self.name, 'occupied': self.occupied}

