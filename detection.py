import numpy as np
from Realsense.realsense_depth import *
from Realsense.realsense import *
import cv2
import torch



class Camera:
    def __init__(self, use_depth_camera):
        self.use_depth_camera = use_depth_camera
        if use_depth_camera:
            self.pipeline = rs.pipeline()
            config = rs.config()

            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)            

            self.pipeline.start(config)
        else:
            self.camera = cv2.VideoCapture(0)

    def __del__(self):
        if self.use_depth_camera:
            self.camera.release()

    def get_current_frame(self):
        if self.use_depth_camera:
            try:
                frames = self.pipeline.wait_for_frames()
            except:
                return False, None, None
                
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if not depth_frame or not color_frame:
                return False, None, None

            return True, depth_image, color_image
        else:
            return True, None, self.camera.read()[1]


class TorchModel:
    def __init__(self):
        self.confidence_threshold = 0.25
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    
    def get_detections(self, color_image):

        detections = self.model(color_image)
        detections_rows = detections.pandas().xyxy

        for i in range(len(detections_rows)):
            detections_rows = detections_rows[i].to_numpy()
        
        return detections_rows



class Detect:
    def __init__(self, use_depth_camera = False):
        self.use_depth_camera = use_depth_camera
        self.camera = Camera(use_depth_camera)
        self.model = TorchModel()
    
    def add_bounding_box(self, color_image, bbxs, label, conf):
        
        x_min, y_min, x_max, y_max = bbxs
        cv2.rectangle(color_image, (int(x_min), int(y_min)), (int( x_max), int(y_max)), (0, 255, 0), 2) 
        return color_image

    def add_depth_label(self, color_image, bbxs, depth):

        x_min, y_min, x_max, y_max = bbxs
        cv2.putText(color_image, "Depth: " + str(depth), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return color_image




    def is_blue(self, color_frame):
        red_lower = np.array([0, 4, 226])
        red_upper = np.array([60, 255, 255])
        blue_lower = np.array([68, 38, 131])
        blue_upper = np.array([113, 255, 255])

        hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(red_contours) > 0 and len(blue_contours) > 0:
            r_area = 0
            b_area = 0
            for c in red_contours:
                r_area += cv2.contourArea(c)
            for c in blue_contours:
                b_area += cv2.contourArea(c)
            if r_area > b_area:
                return False
            else:
                return True
        elif len(red_contours) > 0:
            return False

        return True

    def process_frame(self, color_image, depth_image):

        detections = self.model.get_detections(color_image)

        for i in range(len(detections)):
            if len(detections) > 0:
                x_min, y_min, x_max, y_max, conf, cls, label = detections[i]
                bbox = [x_min, y_min, x_max, y_max]

                if self.is_blue(color_image[int(y_min):int(y_max), int(x_min):int(x_max)]):
                    color_image = self.add_bounding_box(color_image, bbox, label, conf)       

                    coords = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
                    if self.use_depth_camera:
                        color_image = self.add_depth_label(color_image,bbox,depth_image[coords[1],coords[0]])
                    else:
                        color_image = self.add_depth_label(color_image,bbox,"N/A")

        return color_image

    def detect_pipeline(self):
        while True:
            try:
                ret, depth_image, color_image = self.camera.get_current_frame()
            except:
                print("Error getting frame")

            if ret:
                key = cv2.waitKey(1)
                if key == 27:
                    break

                frame = self.process_frame(color_image=color_image, depth_image = depth_image)
                cv2.imshow('RealSense', frame)
                cv2.waitKey(1)


pipeline = Detect()
pipeline.detect_pipeline()