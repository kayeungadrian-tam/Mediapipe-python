import toml
import pprint

import cv2
import yaml
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt




class Config():
    def __init__(self, yaml_file, toml_file) -> None:
        self.yaml_file = yaml_file
        self.toml_file = toml_file
        self._load_yaml()
        self._load_toml()

    def _load_yaml(self) -> None:
        with open(self.yaml_file, "r") as f:
            self.yaml_cfg = yaml.safe_load(f)

    def _load_toml(self) -> None:
        with open(self.toml_file, "r") as f:
            self.toml_cfg = toml.load(f)         


class Detector():
    def __init__(self, thickness, circle_radius, color, min_detection_confidence, min_tracking_confidence) -> None:
        self.drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.drawing.DrawingSpec(
            thickness=thickness, 
            circle_radius=circle_radius, 
            color=color)

        self.drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = mp.solutions.face_mesh
        self.face_detector = self.face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)


    def __call__(self, image):
        self.img = image
        self.results = self.face_detector.process(image)


    def show_detection(self):
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.drawing.draw_landmarks(
                    image=self.img,
                    landmark_list=face_landmarks,
                    connections=self.face_mesh.FACEMESH_TESSELATION,
                    connection_drawing_spec=self.drawing_styles.get_default_face_mesh_tesselation_style()
                    )
        cv2.imshow("detection", self.img)
        cv2.waitKey(0)



class Camera():

    def __init__(self, cfg) -> None:
        self.start(**cfg)

    def start(self, index, fps, frame_width, frame_height) -> None:
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    def load_detector(self, cfg) -> None:
        self.detector = Detector(**cfg)
        # self.detector(img)

    def detect(self, image) -> None:
        self.detector(image)
        self.detector.show_detection()

    def capture(self) -> None:
        while True:
            ret, self.frame = self.cap.read()
            cv2.imshow("demo", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    pass

if __name__ == "__main__":



    img_path = "input/image.jpg"

    src = cv2.imread(img_path)
    # src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    img = src.copy()


    toml_file = "config/realtime.toml"
    yaml_file = "config/makeup.yaml"

    config = Config(yaml_file, toml_file)
    camera_cfg = config.toml_cfg["camera"]
    mediapipe_cfg = config.toml_cfg["mediapipe"]

    CAM = Camera(camera_cfg)
    CAM.load_detector(mediapipe_cfg)    
    # CAM.detect(img)

    CAM.capture()