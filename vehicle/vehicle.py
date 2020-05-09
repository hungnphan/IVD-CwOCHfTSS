import cv2 as cv
import numpy as np
from vehicle.vehicle_properties import Status, VehicleType, TravelingStatus, Direction

class Vehicle(object):
    def __init__(self, boxes, vehicle_image, binary_image,
                 trajectory=None, contours=None, ellipse=None, init_three_feature=True):
        self.zone_index = -1
        self.trajectory_ = [trajectory]
        self.contours_ = [contours]
        self.ellipses_ = [ellipse]
        self.boxes_ = [boxes]
        self.vehicle_images_ = [vehicle_image]
        self.binary_image_ = [binary_image]

        self.vehicle_index = -1
        self.status = Status.Enter
        self.classify_probability = -1
        self.vehicle_type_intcode = -1
        self.vehicle_type = VehicleType.Unidentified
        self.traveling_status = TravelingStatus.Normal
        self.direction = Direction.Downstream
        self.zone_index = -1
        self.frame_rate = 30.0
        self.meter_per_pixel = 0.14444
        self.speed = 0.0
        self.vehicle_sizes_ = list()
        self.dimension_ratios_ = list()
        self.density_ratios_ = list()

        if init_three_feature:
            self.calculate_density_ratio()
            self.calculate_dimension_ratio()
            self.calculate_vehicle_size()

    def calculate_vehicle_10_features(self):

        # Calculate 10 vehicles' features in following order:
        # (0) height_bbox       (1) width_bbox
        # (2) height_ellipse    (3) width_ellipse
        # (4) ellipse_size      (5) vehicle_area
        # (6) cvxhull_size      (7) perimeter_vehicle
        # (8) dimension_ratio   (9) density_ratio
        height_bbox         = self.boxes_[-1][3]
        width_bbox          = self.boxes_[-1][2]
        height_ellipse      = self.ellipses_[-1][1][1]
        width_ellipse       = self.ellipses_[-1][1][0]
        ellipse_size        = height_ellipse * width_ellipse * np.pi/4.0
        vehicle_area        = self.vehicle_sizes_[-1]
        cvxhull_size        = cv.contourArea(cv.convexHull(self.contours_[-1]))
        perimeter_vehicle   = cv.arcLength(self.contours_[-1],True)
        dimension_ratio     = self.dimension_ratios_[-1]
        density_ratio       = self.density_ratios_[-1]

        return np.array([height_bbox,width_bbox,height_ellipse,width_ellipse,ellipse_size,
                         vehicle_area,cvxhull_size,perimeter_vehicle,dimension_ratio,density_ratio])

    def calculate_vehicle_size(self):
        # self.vehicle_sizes_.append(cv.contourArea(self.ellipses_[-1]))
        ellipse = self.ellipses_[-1]
        self.vehicle_sizes_.append(ellipse[1][0]*ellipse[1][1])

    def calculate_dimension_ratio(self):
        rotated_box = cv.minAreaRect(self.contours_[-1])
        width = rotated_box[1][0]
        height = rotated_box[1][1]
        minor_edge = min(width, height)
        major_edge = max(width, height)
        self.dimension_ratios_.append(minor_edge/major_edge)

    def calculate_density_ratio(self):
        object_pixels = cv.countNonZero(self.binary_image_[-1])
        img_width = self.binary_image_[-1].shape[0]
        img_height = self.binary_image_[-1].shape[1]
        total_pixels = 1.0*img_width*img_height
        density_ratio = (1.0*object_pixels)/total_pixels
        self.density_ratios_.append(density_ratio)

    def update_vehicle(self,vehicle_candidate):
        # Update new six features
        # 	Add new: trajectory_ , contours_ , ellipses_
        # 			 boxes_      , vehicle_images_ , binary_image_
        self.update_features(vehicle_candidate)

        # Update new threes classifying features:
        # 	vehicle_size , vehicle_density_ratio , vehicle_dimension_ratio
        # Update moving information of vehicles: status, direction, speed
        self.update_statuses()

        # Update vehicle type
        self.vehicle_type = vehicle_candidate.vehicle_type

    # This function update current vehicle base on vehicle_candidate
    # Function will add the (6) new detected properties:
    #     --| trajectory_	contours_			ellipses_
    #     --| boxes_		vehicle_images_		binary_image_
    def update_features(self, vehicle_candidate):
        self.trajectory_.append(vehicle_candidate.trajectory_[-1])
        self.contours_.append(vehicle_candidate.contours_[-1])
        self.ellipses_.append(vehicle_candidate.ellipses_[-1])
        self.boxes_.append(vehicle_candidate.boxes_[-1])
        self.vehicle_images_.append(vehicle_candidate.vehicle_images_[-1])
        self.binary_image_.append(vehicle_candidate.binary_image_[-1])

    def update_statuses(self):
        # Calculate classification features
        self.calculate_vehicle_size()
        self.calculate_dimension_ratio()
        self.calculate_density_ratio()

        # Update status, direction, speed, and vehicle type
        self.update_moving_status()
        self.update_moving_direction()
        self.calculate_moving_speed()

    def update_moving_status(self):
        frame_count = int(len(self.trajectory_))
        if frame_count <= 1:
            self.status_ = Status.Enter
        elif (1 < frame_count <= 2):
            self.status_ = Status.Validating
        elif (frame_count > 2 and self.status_ == Status.Validating):
            self.status_ = Status.Classifying

    def update_moving_direction(self):
        if self.status_ == Status.Validating \
           or self.status_ == Status.Classifying:
            vertical_displacement = float(self.trajectory_[-1][1] - self.trajectory_[0][1])
            if vertical_displacement > 0:
                direction_ = Direction.Downstream
            else:
                direction_ = Direction.Upstream

    def calculate_moving_speed(self):
        if self.status == Status.Classifying:
            self.speed = 0.0
        else:
            # 1 m/s = 3.6 km.h
            converted_speed = self.calculate_vehicle_pixel_speed() * self.meter_per_pixel
            if (converted_speed < 1.0):
                self.speed = 0.0
            else:
                self.speed = converted_speed

    def calculate_vehicle_pixel_speed(self):
        x_displacement = abs(self.trajectory_[-1][0] - self.trajectory_[0][0])
        y_displacement = abs(self.trajectory_[-1][1] - self.trajectory_[0][1])
        pixel_displacement = np.sqrt(1.0*x_displacement*x_displacement + 1.0*y_displacement*y_displacement)
        return (pixel_displacement * self.frame_rate * 3.6) / (1.0*len(self.trajectory_))

    def add_trajectory(self, trajectory):
        self.trajectory_.append(trajectory)

    def add_add_contour(self, contour):
        self.contours_.append(contour)

    def add_ellipse(self, ellipse):
        self.ellipses_.append(ellipse)

    def add_box(self, box):
        self.boxes_.append(box)

    def add_vehicle_image(self, vehicle_image):
        self.vehicle_images_.append(vehicle_image)

    def add_binary_image(self, binary_image):
        self.binary_image_.append(binary_image)

    def add_vehicle_size(self, vehicle_size):
        self.vehicle_sizes_.append(vehicle_size)

    def add_dimenstion_ratio(self, dimension_ratio):
        self.dimension_ratios_.append(dimension_ratio)

    def add_density_ratio(self, density_ratio):
        self.density_ratios_.append(density_ratio)



