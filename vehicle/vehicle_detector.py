import cv2 as cv
from vehicle.vehicle import Vehicle
import numpy as np

class VehicleDetector(object):
    def __init__(self):
        self.kMinContourSize = 5
        self.kMinVehicleSize = 100.0
        self.kMaxVehicleSize = 50000.0

    def detect_vehicle_candidate(self, foreground_img, background_img, rbg_image):
        contours = self.extract_contours(foreground_img)

        vehicles_ = list()
        for contour_ in contours:
            ## Skip the small pieces ~ consider as noises
            if (self.is_valid_contour(contour_) == False):
                continue

            ## Construct new vehicle features
            ellipse = cv.fitEllipse(contour_)                   # @ellipse      : ((x,y),(w,h),angle)
            box = cv.boundingRect(contour_)                     # @box          : (x,y,w,h)
            trajectory = tuple([int(ellipse[0][0]),int(ellipse[0][1])])

            vehicle_image = rbg_image[box[1]:box[1]+box[3],     # x = box[0]    y = box[1]
                                      box[0]:box[0]+box[2]]     # w = box[2]    h = box[3]
            binary_image = foreground_img[box[1]:box[1]+box[3],
                                          box[0]:box[0]+box[2]]

            ## Create a vehicle_candidate and add to returned list
            vehicle_candidate_ = Vehicle(trajectory=trajectory,
                                         contours=contour_,
                                         ellipse=ellipse,
                                         boxes=box,
                                         vehicle_image=vehicle_image,
                                         binary_image=binary_image,
                                         init_three_feature=True)
            vehicles_.append(vehicle_candidate_)

        return vehicles_

    def extract_contours(self, foreground_img):
        ## Output contains 3 parts:
        ## modified image, the contours and hierarchy
        im2, contours_, hierarchy = cv.findContours(foreground_img,
                                                     cv.RETR_EXTERNAL,
                                                     cv.CHAIN_APPROX_SIMPLE)
        return contours_

    def is_valid_contour(self, contour):
        if(contour.shape[0] >= self.kMinContourSize):
            contour_area_ = cv.contourArea(contour)
            if(self.kMinVehicleSize <= contour_area_ <= self.kMaxVehicleSize):
                return True
            return False
        else:
            return False


