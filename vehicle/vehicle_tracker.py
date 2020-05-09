from vehicle.vehicle_properties import Status
import numpy as np

class VehicleTracker(object):
    def __init__(self):
        self.kLookbackLimit = 6
        self.kMaxHorizontalDistance = 2
        self.kMaxVerticalDistance = 8.0
        self.kMaxSizeRatio = 0.6

    def track_vehicles(self, vehicle_candidates, vehicles):
        # Case 1: no vehicle candidate is detected --> no vehicles
        if len(vehicle_candidates) == 0:
            vehicles.clear()
        else:
            # Case 2: vehicles is not empty
            # --> match vehicle candidates with previous vehicles
            if len(vehicles) != 0:
                vehicle_candidates, vehicles = self.match_vehicles(vehicle_candidates, vehicles)

            # Both cases 2 and 3: add vehicle candidates that are not matched as new vehicle
            for vc in vehicle_candidates:
                if vc.status == Status.Enter:
                    vehicles.append(vc)
            vehicle_candidates = self.deleteMatchedVehicleCandidates(vehicle_candidates)
            vehicles = self.deleteExitVehicles(vehicles)

        return vehicle_candidates, vehicles

    def match_vehicles(self, vehicle_candidates, vehicles):
        for v in vehicles:
            vechicle_trajectory = v.trajectory_
            vehicle_sizes = v.vehicle_sizes_
            is_matched = False
            is_exiting = True

            for vc in vehicle_candidates:
                if is_matched:
                    break
                if not vc.status == Status.Enter:
                    continue

                # We check back at most kLookbackLimit frames
                for cnt in range(0,min(self.kLookbackLimit,len(v.trajectory_)),1):
                    t = cnt+1
                    k = -1 - cnt

                    horizontal_distance = self.horizontalDistance(vechicle_trajectory[k],vc.trajectory_[-1])
                    vertical_distance   = self.verticalDistance(vechicle_trajectory[k],vc.trajectory_[-1])
                    size_ratio = self.sizeRatio(vehicle_sizes[k], vc.vehicle_sizes_[-1])

                    if (horizontal_distance <= self.kMaxHorizontalDistance * t      # condition 1
                    and vertical_distance   <= self.kMaxVerticalDistance * t        # condition 2
                    and size_ratio >= self.kMaxSizeRatio):
                        v.update_vehicle(vc)        # update vehicle_[i]
                        vc.status = Status.Exit     # mark vehicle_candidate_[j] for deletion
                        is_matched = True
                        is_exiting = False
                        break
            # Can't find a match --> mark this vehicle for deletion later
            if is_exiting:
                v.status = Status.Exit
        return vehicle_candidates, vehicles

    def deleteExitVehicles(self, vehicles):
        del_idx = list()
        for idx, vehicle in enumerate(vehicles):
            if vehicle.status == Status.Exit:
                del_idx.append(idx)
        return [vehicle
                for idx, vehicle in enumerate(vehicles)
                if idx not in del_idx]

    def deleteMatchedVehicleCandidates(self, vehicle_candidates):
        del_idx = list()
        for idx, vehicle in enumerate(vehicle_candidates):
            if vehicle.status == Status.Exit:
                del_idx.append(idx)
        return [vehicle
                for idx, vehicle in enumerate(vehicle_candidates)
                if idx not in del_idx]

    def horizontalDistance(self, vehicle_centroid, candidate_centroid):
        return abs(vehicle_centroid[0] - candidate_centroid[0])

    def verticalDistance(self, vehicle_centroid, candidate_centroid):
        return abs(vehicle_centroid[1] - candidate_centroid[1])

    def euclideanDistance(self, vehicle_centroid, candidate_centroid):
        return np.sqrt(
                    ((vehicle_centroid[1] - candidate_centroid[1]) * (vehicle_centroid[1] - candidate_centroid[1]))
                    + ((vehicle_centroid[0] - candidate_centroid[0]) * (vehicle_centroid[0] - candidate_centroid[0]))
                )

    def sizeRatio(self, vehicle_size, candidate_size):
        return min(vehicle_size,candidate_size)/max(vehicle_size,candidate_size)

