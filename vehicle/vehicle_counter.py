from enum import Enum
from vehicle.vehicle_properties import VehicleType

class VehicleCounter(object):
    def __init__(self):
        self.vehicle_counts_ = {
            VehicleType.Class1 : 0,
            VehicleType.Class2 : 0,
            VehicleType.Class3 : 0,
            VehicleType.Unidentified : 0,
            VehicleType.Vehicle_Block : 0
        }
        self.total_vehicle = 0
        self.total_speed_ = 0.0
        self.avg_speed_ = 0.0

    def update_vehicle_count(self,vehicle_type):
        self.vehicle_counts_[vehicle_type]+=1

    def update_vehicle_speed(self, vehicle_speed):
        self.total_speed_ += vehicle_speed
        total_vehicle_count = self.vehicle_counts_[VehicleType.Class1] \
                              + self.vehicle_counts_[VehicleType.Class2] \
                              + self.vehicle_counts_[VehicleType.Class3]
        self.avg_speed_ = 1.0*self.total_speed_ /(1.0*total_vehicle_count)


