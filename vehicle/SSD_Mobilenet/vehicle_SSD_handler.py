import cv2 as cv
import numpy as np
from vehicle.vehicle import Vehicle
from vehicle.vehicle_properties import VehicleType
from vehicle.vehicle_detector import VehicleDetector

class VehicleOcclusionHandler(object):
    def __init__(self):
        self.tf_network                     = None
        self.g_width                        = 80
        self.g_height                       = 80
        self.g_mean_val                     = 127.5
        self.g_scale_factor                 = 1.0 / self.g_mean_val
        self.g_confidence_threshold         = 0.5
        self.ssd_graph_pb                   = "vehicle/SSD_Mobilenet/frozen_inference_graph.pb"
        self.ssd_graph_pbtxt                = "vehicle/SSD_Mobilenet/frozen_inference_graph.pbtxt"

        self.vehicle_lable_map = [VehicleType.Vehicle_Block,
                                  VehicleType.Class1,
                                  VehicleType.Class2,
                                  VehicleType.Class3]

        self.__initialize_SSD_MobileNet__()
        self.vehicle_detector = VehicleDetector()

    def __initialize_SSD_MobileNet__(self):
        self.tf_network = cv.dnn.readNetFromTensorflow(self.ssd_graph_pb,
                                                       self.ssd_graph_pbtxt)

    def handle_occlusion_blob(self, occlusion_blobs, rbg_image, foreground_img):
        extracted_vehicles = list()

        # Prepare a list of blobs to pass once into SSD model
        image_blobs = []
        for blob in occlusion_blobs:
            image_blobs.append(blob.vehicle_images_[-1])

        input_blob = cv.dnn.blobFromImages(image_blobs,
                                           self.g_scale_factor,
                                           (self.g_width, self.g_height),
                                           (self.g_mean_val, self.g_mean_val, self.g_mean_val),
                                           swapRB=True,
                                           crop=False)
        self.tf_network.setInput(input_blob)
        detection = self.tf_network.forward()

        for weight in detection[0, 0, :, :]:
            img_idx = int(weight[0])
            blob = occlusion_blobs[img_idx]
            object_class = weight[1]
            score = float(weight[2])

            origin_anchor = (blob.boxes_[-1][0],blob.boxes_[-1][1])   # (x,y)

            if score > self.g_confidence_threshold:
                left    = weight[3] * blob.vehicle_images_[-1].shape[1] + origin_anchor[0]
                top     = weight[4] * blob.vehicle_images_[-1].shape[0] + origin_anchor[1]
                right   = weight[5] * blob.vehicle_images_[-1].shape[1] + origin_anchor[0]
                bottom  = weight[6] * blob.vehicle_images_[-1].shape[0] + origin_anchor[1]

                # Construct extracted vehicle from occlusion blob
                box = (int(left),int(top),int(right-left),int(bottom-top))      # @box : (x,y,w,h)
                vehicle_image = rbg_image[box[1]:box[1]+box[3],                 # x = box[0]    y = box[1]
                                          box[0]:box[0]+box[2]]                 # w = box[2]    h = box[3]
                binary_image = foreground_img[box[1]:box[1]+box[3],
                                              box[0]:box[0]+box[2]]

                # [Option 1]: Full version
                contours = self.vehicle_detector.extract_contours(binary_image)
                contour_ = contours[0]
                contour_ = contour_ + np.array([[origin_anchor[0],origin_anchor[1]]])

                if (self.vehicle_detector.is_valid_contour(contour_) == False):
                    continue

                ## Construct new vehicle features
                ellipse = cv.fitEllipse(contour_)           # @ellipse      : ((x,y),(w,h),angle)
                trajectory = tuple([int(ellipse[0][0]), int(ellipse[0][1])])

                ## Create a vehicle_candidate and add to returned list
                extracted_vehicle = Vehicle(trajectory=trajectory,
                                             contours=contour_,
                                             ellipse=ellipse,
                                             boxes=box,
                                             vehicle_image=vehicle_image,
                                             binary_image=binary_image,
                                             init_three_feature=True)

                # [Option 2]: Short version
                # extracted_vehicle = Vehicle(boxes=box,
                #                             vehicle_image=vehicle_image,
                #                             binary_image=binary_image,
                #                             init_three_feature=False)

                # Set detection confidence and vehicle type
                extracted_vehicle.classify_probability = score
                extracted_vehicle.vehicle_type = self.vehicle_lable_map[int(object_class)]
                extracted_vehicles.append(extracted_vehicle)

        return extracted_vehicles

