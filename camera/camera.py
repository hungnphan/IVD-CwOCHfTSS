import cv2 as cv
import os
import time
from data_io.data_loader import DataLoader
from data_io.io_util import IOUtil
from vehicle.vehicle_properties import Status,TravelingStatus
from vehicle.vehicle_detector import VehicleDetector
from vehicle.vehicle_classifier import VehicleClassifier
from vehicle.SSD_Mobilenet.vehicle_SSD_handler import VehicleOcclusionHandler
from vehicle.vehicle_tracker import VehicleTracker
from vehicle.vehicle_counter import VehicleCounter

class Camera(object):
    def __init__(self, dataset_name):
        self.data_dir = "data_"
        #self.dataset_name = "PVD01"
        self.dataset_name = dataset_name   
        self.observation_zones = list()
        self.cap_bg = None
        self.cap_fg = None
        self.cap_im = None
        self.vehicle_candidates_ = list()
        self.vehicles_ = list()

        ## Initialize input and output streams of camera
        # IOUtil.print_warning_messages("Initializing Camera Input-Output stream ... ")
        # self.setup_camera_stream()

        ## Initialize oz config from config files
        IOUtil.print_warning_messages("Initializing observation zones from config file ... ")
        self.initialize_observation_zone()

        ## Initialize vehicle detector
        IOUtil.print_warning_messages("Initializing vehicle detector ... ")
        self.initialize_vehicle_detector()

        ## Initialize vehicle classifier
        IOUtil.print_warning_messages("Initializing vehicle classifier ... \n")
        self.initialize_vehicle_classifier()

        # Initialize vehicle occlusion handler
        self.initialize_vehile_occlusion_handler()

        ## Initialize vehicle tracker
        self.initialize_vehicle_tracker()
        self.initialize_vehicle_counter()

        ###################################################
        ## Main process is here (^.^)
        ###################################################
        IOUtil.print_warning_messages("Camera is ready to process ... ")
        self.run()

    #################################################
    ## Initializing functions
    #################################################
    def setup_camera_stream(self):
        data_loader = DataLoader(data_dir=self.data_dir,
                                 dataset_name=self.dataset_name)
        self.cap_bg, self.cap_fg, self.cap_im = data_loader.init_input_stream()

    def initialize_observation_zone(self):
        ## read from config file and construct into ObservationZone objects
        self.observation_zones = IOUtil.load_observation_zone_config(self.data_dir + "/" + self.dataset_name + "/" + "config.txt")

    def initialize_vehicle_detector(self):
        ## initialize VehicleDetector from list of ObservationZone objects
        self.vehicle_detector = VehicleDetector()

    def initialize_vehicle_classifier(self):
        ## initialize VehicleClassifier from Neural Decision Tree
        self.vehicle_classifier = VehicleClassifier(max_tree_depth = 7,
                                                    n_features = 10,
                                                    n_classes = 4)

    def initialize_vehile_occlusion_handler(self):
        self.vehicle_occlusion_handler = VehicleOcclusionHandler()

    def initialize_vehicle_tracker(self):
        self.vehicle_tracker = VehicleTracker()

    def initialize_vehicle_counter(self):
        self.vehicle_counter_ = VehicleCounter()

    #################################################
    ## Detect and Classify vehicles
    #################################################
    def detect_vehicle(self, foreground_img, background_img, rbg_image):
        detected_candidate = self.vehicle_detector.detect_vehicle_candidate(foreground_img,
                                                                            background_img,
                                                                            rbg_image)
        ## Filter noises and remove unncessary candidates
        del_idx = list()
        for idx, vehicle in enumerate(detected_candidate):
            found = False

            for oz_idx, observation_zone in enumerate(self.observation_zones):
                check_status = observation_zone.check_inside_oz_and_set_oz_index(vehicle)
                if check_status:
                    vehicle.zone_index = observation_zone.oz_index
                    found = True
                    break
            if not found:       # If not found any OZ containing the vehicle
                del_idx.append(idx)

        filter_detected_candidate = [vehicle
                                     for idx, vehicle in enumerate(detected_candidate)
                                        if idx not in del_idx]
        return filter_detected_candidate

    def classify_vehicle(self, detected_vehicles):
        vehicles, blobs = self.vehicle_classifier.detect_occlusion(detected_vehicles)

        if not len(vehicles) == 0:
            self.vehicle_classifier.classifiy_vehicles(vehicles)

        return vehicles, blobs

    #################################################
    ## Tracking and Counting vehicles
    #################################################
    def track_and_count_vehicle(self, vehicle_candidates, vehicles):
        vehicle_candidates, vehicles = self.track_vehicle(vehicle_candidates, vehicles)
        self.count_vehicle(vehicles)
        return vehicle_candidates, vehicles

    def track_vehicle(self, vehicle_candidates, vehicles):
        return self.vehicle_tracker.track_vehicles(vehicle_candidates, vehicles)

    def count_vehicle(self, vehicles_):
        for v in vehicles_:
            zone_index=v.zone_index

            if(v.status == Status.Classifying and zone_index > -1):
                if(self.observation_zones[zone_index].isViolatedVehicle(v)):
                    v.traveling_status = TravelingStatus.WrongWayDriving
                else:
                    v.traveling_status = TravelingStatus.Normal

                # Count vehicle
                if(self.observation_zones[zone_index].isVehicleCountable(v)):
                    self.vehicle_counter_.update_vehicle_count(v.vehicle_type)
                    self.vehicle_counter_.update_vehicle_speed(v.speed)
                    v.status = Status.Counted


    #################################################
    ## Drawing & Displaying functions of camera
    #################################################
    def draw_result(self, rbg_image, vehicles):
        self.draw_observation_zone(rbg_image)
        self.draw_vehicle_boundary(rbg_image, vehicles)
        self.draw_trajectory(rbg_image, vehicles)

    def draw_observation_zone(self, rbg_image):
        IOUtil.draw_observation_zone(rbg_image, self.observation_zones)

    def draw_vehicle_boundary(self, rbg_image, vehicles):
        # IOUtil.draw_vehicle_bounding_ellipse(rbg_image, vehicles)
        IOUtil.draw_vehicle_bounding_rectangle(rbg_image, vehicles)
        # IOUtil.draw_lable_and_prob_vehicle(rbg_image, vehicles)

    def draw_trajectory(self, rbg_image, vehicles):
        for vidx, vehicle in enumerate(vehicles):
            trajectory = vehicle.trajectory_

            # print("--> vehicle",vidx,":",len(trajectory))

            for idx in range(len(trajectory)-1):
                point_1 = trajectory[idx]
                point_2 = trajectory[(idx+1)]
                cv.line(rbg_image, point_1, point_2, (0,255,0), 1)

    def show_result_windows(self, foreground_img, background_img, rbg_image):
        IOUtil.show_background(background_img)
        IOUtil.show_foreground(foreground_img)
        IOUtil.show_result(rbg_image)

    #################################################
    ## Main processing of camera
    #################################################
    def run(self):
        start_time = time.time()
        nFrame = 0

        out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc('M','J','P','G'), 30, (640,360))
        ########################################################################################
        # Read dataset:
        ########################################################################################

        ## [Option 1] - Read from frames
        for image_idx in os.listdir(self.data_dir + '/' + self.dataset_name + '/im'):
            nFrame = nFrame + 1

            input_image = cv.imread(self.data_dir + '/' + self.dataset_name + '/im/' + image_idx)
            background = cv.imread(self.data_dir + '/' + self.dataset_name + '/bg/' + image_idx)
            foreground = cv.imread(self.data_dir + '/' + self.dataset_name + '/fg/' + image_idx,
            					   cv.IMREAD_GRAYSCALE)
            
            if input_image is not None and background is not None and foreground is not None:

        ## [Option 2] - Read from videos
        # while(True):
        #     nFrame = nFrame + 1
        #     ret_im, input_image = self.cap_im.read()
        #     ret_bg, background = self.cap_bg.read()
        #     ret_fg, foreground = self.cap_fg.read()

        #     if ret_im == True and ret_bg == True and ret_fg == True:
                # Convert foreground to GRAY mode
                # foreground = cv.cvtColor(foreground,cv.COLOR_BGR2GRAY)

        ########################################################################################
        # Main process:
        ########################################################################################

                # IOUtil.print_fps_message("     |--- frame #" + str(counter))

                ## Vehicle detection & Extract features
                detected_vehicle_candidate = self.detect_vehicle(foreground,background,input_image)

                ## Vehicle classification using Neural Decision Tree
                vehicles, blobs = self.classify_vehicle(detected_vehicle_candidate)

                ## Vehicle occlusion handling using SSD-MobileNet
                if not len(blobs) == 0:
                    extracted_vehicles = self.vehicle_occlusion_handler.handle_occlusion_blob(blobs,
                                                                                              input_image,
                                                                                              foreground)
                    detected_vehicles = vehicles + extracted_vehicles
                else:
                    detected_vehicles = vehicles

                ## Track and count vehicles:
                self.vehicle_candidates_ = detected_vehicles
                # self.vehicle_candidates_, self.vehicles_ = self.track_and_count_vehicle(self.vehicle_candidates_,
                #                                                                         self.vehicles_)

                ## Draw results
                self.draw_result(input_image, self.vehicle_candidates_)
                # self.draw_result(input_image, detected_vehicle_candidate)

                ## Show result
                self.show_result_windows(foreground, background, input_image)
                out.write(input_image)
                ## [Option 1]: Output frame-by-frame
                # cv.waitKey(0);

                ## [Option 2]: Output continously a sequence of frame
                c = cv.waitKey(1)
                if (c == 27):
                    break
                elif (c == 32):
                    while (cv.waitKey(0) != 32):
                        continue

                # cv.imwrite("F:/export/"+ self.dataset_name + "/" + image_idx,input_image)

                end_time = time.time()
                # print("--- %s seconds ---" % (end_time - start_time))
                # print("nFrame =",nFrame)
                #IOUtil.print_fps_message("     |--- fps =" + str(1.0*nFrame/(end_time - start_time)))
            else:
                # [Option 1 - RUN ONCE]: if out of frame then stop the process
                break

                # [Option 2 - LOOP REWIND]: if out of frame then reset to the beginning
                # self.cap_bg.set(cv.CAP_PROP_POS_MSEC, 0)
                # self.cap_fg.set(cv.CAP_PROP_POS_MSEC, 0)
                # self.cap_im.set(cv.CAP_PROP_POS_MSEC, 0)

        cv.destroyAllWindows()
        out.release()


    

