import cv2 as cv
from vehicle.vehicle import Vehicle
from vehicle.vehicle_properties import Direction

class ObservationZone(object):
    def __init__(self, oz_index, direction, oz_region):
        self.oz_index = oz_index
        self.direction = direction
        self.oz_region = oz_region
        self.vehicle_candidates = list()
        self.vehicles = list()

    def check_inside_oz_and_set_oz_index(self, vehicle):
        oz_status = cv.pointPolygonTest(self.oz_region,
                                        vehicle.trajectory_[-1],
                                        False)

        if(oz_status>=0):
            return True
        else:
            return False

    def isViolatedVehicle(self, vehicle):
        return vehicle.direction == self.direction

    def isVehicleCountable(self, vehicle):
        vehicle_centroid_y = vehicle.ellipses_[-1][0][1]
        mid_left_point_y = self.oz_region[5][1]
        if (self.direction == Direction.Downstream):
            return (vehicle_centroid_y < mid_left_point_y + 30) \
                   and (vehicle_centroid_y > mid_left_point_y)
        elif (self.direction == Direction.Upstream):
            return (vehicle_centroid_y > mid_left_point_y - 30) \
                   and (vehicle_centroid_y < mid_left_point_y)
        return False














