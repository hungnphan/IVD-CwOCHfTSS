import cv2 as cv
import numpy as np
import colorama
from camera.observation_zone import ObservationZone
from data_io.common import Color
from data_io.common import ConsoleColor
from vehicle.vehicle_properties import VehicleColor,Direction

class IOUtil(object):
    def __init__(self):
        pass

    @staticmethod
    def print_warning_messages(message):
        colorama.init(autoreset=True)
        print(ConsoleColor.Green.value + message)

    @staticmethod
    def print_fps_message(message):
        colorama.init(autoreset=True)
        print(ConsoleColor.Red.value + message)

    @staticmethod
    def load_observation_zone_config(path_to_config_file):
        config_file = open(path_to_config_file)

        # Read run_mode
        line = config_file.readline()

        # Read number of observation_zone
        line = config_file.readline()
        token = line.split()
        nObservationZone = int(token[1])

        observation_zones = list()
        for oz_idx in range(nObservationZone):
            # Skip header line
            line = config_file.readline()

            # Read direction of traffic flow
            line = config_file.readline()
            token = line.split()
            if token == "down":
                direction = Direction.Downstream
            else:
                direction = Direction.Upstream

            # Read number of vertices of oz region
            line = config_file.readline()
            token = line.split()
            nPoint = int(token[1])

            ## Read oz_region_ coordinates in this order:
            ##      (0,top_left)-------------(1,top_right)
            ##       |                             |
            ##     (5,middle_left)          (2,middle_right)
            ##       |                             |
            ##     (4,bottom_left)----------(3,bottom_right)
            oz_region = list()
            for vertice_idx in range(nPoint):
                line = config_file.readline()
                token = line.split()
                x = int(token[1])
                y = int(token[2])
                oz_region.append((x,y))

            # If there is only 4 points, add mid-points and convert into 6 points
            if nPoint==4:
                oz_region.append(oz_region[3])
                oz_region[3] = oz_region[2]
                oz_region[2] = (int((oz_region[1][0] + oz_region[3][0]) / 2.0),
                                int((oz_region[1][1] + oz_region[3][1]) / 2.0))
                oz_region.append((int((oz_region[0][0] + oz_region[4][0]) / 2.0),
                                  int((oz_region[0][1] + oz_region[4][1]) / 2.0)))

            oz_region = np.array(oz_region)
            observation_zones.append(ObservationZone(oz_index=oz_idx,
                                                     direction=direction,
                                                     oz_region=oz_region))

            # Read and skip the vehicle constraints
            for vehicle_type in range(3):
                # Skip a blank line
                line = config_file.readline()
                # Skip header line of each vehicle type
                line = config_file.readline()
                # Read vehicle size
                line = config_file.readline()
                # Read vehicle dimension
                line = config_file.readline()
                # Read vehicle density
                line = config_file.readline()

            # Skip a blank line at the end of each observation zone
            line = config_file.readline()

        return observation_zones

    @staticmethod
    def show_background(background_img):
        cv.imshow("Background", background_img)

    @staticmethod
    def show_foreground(foreground_img):
        cv.imshow("Foreground", foreground_img)

    @staticmethod
    def show_result(result_img):
        cv.imshow("Result", result_img)

    @staticmethod
    def draw_observation_zone(output_frame, observation_zones):
        for oz in observation_zones:
            oz_size = len(oz.oz_region)
            for oz_point_idx in range(oz_size):
                next_idx = (oz_point_idx+1)%oz_size

                cv.line(output_frame,
                        tuple(oz.oz_region[oz_point_idx]),
                        tuple(oz.oz_region[next_idx]),
                        Color.Green.value, 2 )

    @staticmethod
    def draw_vehicle_bounding_rectangle(output_frame, vehicles):
        for vehicle in vehicles:
            rect = vehicle.boxes_[-1]
            cv.rectangle(output_frame,
                         tuple([rect[0], rect[1]]),
                         tuple([rect[0] + rect[2], rect[1] + rect[3]]),
                         VehicleColor[vehicle.vehicle_type.value], 2)

    @staticmethod
    def draw_vehicle_bounding_ellipse(output_frame, vehicles):
        for vehicle in vehicles:
            rect = vehicle.boxes_[-1]
            cv.ellipse(output_frame,
                       vehicle.ellipses_[-1],
                       VehicleColor[vehicle.vehicle_type.value], 2)

    @staticmethod
    def draw_lable_and_prob_vehicle(output_frame, vehicles):
        for vehicle in vehicles:
            box = vehicle.boxes_[-1]
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])

            text = "{}:{:.4f}".format(vehicle.vehicle_type_intcode,
                                       vehicle.classify_probability)
            cv.putText(output_frame,
                       text,
                       (x, y - 5),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5, VehicleColor[vehicle.vehicle_type.value], 2)

