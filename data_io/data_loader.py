import cv2 as cv


class DataLoader(object):
    def __init__(self, data_dir, dataset_name):
        self.data_dir = data_dir
        self.dataset_name = dataset_name

    def init_input_stream(self):
        # cap_bg = cv.VideoCapture(self.data_dir+"/" +
        #                          self.dataset_name +
        #                          "/bg/"+"%07d.png", cv.CV_CAP_FFMPEG )
        # cap_fg = cv.VideoCapture(self.data_dir+"/" +
        #                          self.dataset_name +
        #                          "/fg/"+"%07d.png", cv.CV_CAP_FFMPEG )
        # cap_im = cv.VideoCapture(self.data_dir+"/" +
        #                          self.dataset_name +
        #                          "/im/"+"%07d.png", cv.CV_CAP_FFMPEG )

        cap_bg = cv.VideoCapture(self.data_dir+"/" +
                                 self.dataset_name + "/"
                                 "bg.mp4")
        cap_fg = cv.VideoCapture(self.data_dir+"/" +
                                 self.dataset_name + "/"
                                 "fg.mp4")
        cap_im = cv.VideoCapture(self.data_dir+"/" +
                                 self.dataset_name + "/"
                                 "im.mp4")

        return cap_bg, cap_fg, cap_im


