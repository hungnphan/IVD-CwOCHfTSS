from vehicle.neural_decision_tree.model import SoftDecisionTree
from vehicle.vehicle_properties import VehicleType
import tensorflow as tf
import numpy as np
import cv2 as cv

class VehicleClassifier(object):
    def __init__(self, max_tree_depth, n_features, n_classes, max_leafs = None):
        self.max_tree_depth = max_tree_depth
        self.n_features = n_features
        self.n_classes = n_classes
        self.max_leafs = max_leafs

        # Define sess
        tf.reset_default_graph()
        self.sess = tf.Session()

        # Declare the graph of default Neural Decision Tree
        self.init_neural_decision_tree()

        # Initialize the variable tensor for Neural Decision Tree
        self.init_tree_variables()

        # Load the trained model from checkpoints
        self.saver = None
        self.load_model_from_checkpoint()

        self.occlusion_thres = 120

    def init_neural_decision_tree(self):
        self.neural_decision_tree = SoftDecisionTree(max_depth=self.max_tree_depth,
                                                     n_features=self.n_features,
                                                     n_classes=self.n_classes,
                                                     max_leafs=self.max_leafs)
        self.neural_decision_tree.build_tree()

    def init_tree_variables(self):
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def load_model_from_checkpoint(self):
        checkpoint_dir = "vehicle/neural_decision_tree/log/"
        self.saver = tf.train.Saver(max_to_keep=5)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ... ".format(latest_checkpoint),end="")
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded\n")
        else:
            print("No checkpoint found !!")

    # This is the primary function called in Camera
    # to classify a list of vehicles
    def classifiy_vehicles(self, vehicles):
        vehicle_features = self.extract_vehicle_feature_vector(vehicles)
        classification = self.classifiy_by_neural_decision_tree(vehicle_features)
        self.lable_vehicle(vehicles, classification)

    # Separate occlusion blobs and vehicle candidates
    def detect_occlusion(self, vehicles):
        vehicle_candidates = list()
        vehicle_overlap = list()

        for vehicle in vehicles:
            vehicle_area            = cv.contourArea(vehicle.contours_[-1])
            vehicle_convexhull_area = cv.contourArea(cv.convexHull(vehicle.contours_[-1]))
            diff = vehicle_convexhull_area - vehicle_area

            if diff >= self.occlusion_thres:
                vehicle.vehicle_type = VehicleType.Vehicle_Block
                vehicle.vehicle_type_intcode = 0
                vehicle_overlap.append(vehicle)
            else:
                vehicle_candidates.append(vehicle)

        return vehicle_candidates, vehicle_overlap

    def extract_vehicle_feature_vector(self, vehicles):
        vehicle_features = list()
        for vehicle in vehicles:
            vehicle_features.append(vehicle.calculate_vehicle_10_features())
        return np.array(vehicle_features)

    def classifiy_by_neural_decision_tree(self, vehicle_features):
        classification = self.sess.run(self.neural_decision_tree.final_output,
                                       feed_dict={self.neural_decision_tree.tf_X: vehicle_features})
        return classification

    def lable_vehicle(self, vehicles, classification_result):
        ## classification_result has shape (nVehicles,2)

        vehicle_lable_map = [VehicleType.Vehicle_Block,
                             VehicleType.Class1,
                             VehicleType.Class2,
                             VehicleType.Class3]

        for vehicle,cls in zip(vehicles,classification_result):
            vehicle.classify_probability = cls[0]
            vehicle.vehicle_type = vehicle_lable_map[int(cls[1])]
            vehicle.vehicle_type_intcode = int(cls[1])

    

