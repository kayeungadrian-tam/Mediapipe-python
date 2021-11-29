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
            static_image_mode=False,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

        self._parts_init()
        
    def __call__(self, image):
        self.img = image
        self.results = self.face_detector.process(image)
        return self.results

    def _parts_init(self) -> None:
        self.lips = list(self.face_mesh.FACEMESH_LIPS)
        self.lips = np.ravel(self.lips)
        
        self.l_eyes = list(self.face_mesh.FACEMESH_LEFT_EYE)
        self.l_eyes = np.ravel(self.l_eyes)
        
        self.r_eyes = list(self.face_mesh.FACEMESH_RIGHT_EYE)
        self.r_eyes = np.ravel(self.r_eyes)
        
        self.l_eyebrow = list(self.face_mesh.FACEMESH_LEFT_EYEBROW)
        self.l_eyebrow = np.ravel(self.l_eyebrow)
        
        self.r_eyebrow = list(self.face_mesh.FACEMESH_RIGHT_EYEBROW)
        self.r_eyebrow = np.ravel(self.r_eyebrow)

    def post_processing(self, mask, cfg):

        face_dict = {}
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:


                mask_lip = []
                mask_face = []
                mask_l_eyes = []
                mask_r_eyes = []
                mask_l_eyebrow = []
                mask_r_eyebrow = []
                for i in range(self.face_mesh.FACEMESH_NUM_LANDMARKS):
                    
                    if i in self.lips:
                        pt1 = face_landmarks.landmark[i]
                        x = int(pt1.x * self.img.shape[1])
                        y = int(pt1.y * self.img.shape[0])
                        mask_lip.append((x, y))

                        
                    elif i in self.l_eyes:
                        pt1 = face_landmarks.landmark[i]
                        x = int(pt1.x * self.img.shape[1])
                        y = int(pt1.y * self.img.shape[0])

                        mask_l_eyes.append((x, y))
                    
                    elif i in self.r_eyes:
                        pt1 = face_landmarks.landmark[i]
                        x = int(pt1.x * self.img.shape[1])
                        y = int(pt1.y * self.img.shape[0])
                        mask_r_eyes.append((x, y))
                    
                    elif i in self.r_eyebrow:
                        pt1 = face_landmarks.landmark[i]
                        x = int(pt1.x * self.img.shape[1])
                        y = int(pt1.y * self.img.shape[0])
                        mask_r_eyebrow.append((x, y))
                    
                    elif i in self.l_eyebrow:
                        pt1 = face_landmarks.landmark[i]
                        x = int(pt1.x * self.img.shape[1])
                        y = int(pt1.y * self.img.shape[0])
                        mask_l_eyebrow.append((x, y))
                    
                    else:
                        pt1 = face_landmarks.landmark[i]
                        x = int(pt1.x * self.img.shape[1])
                        y = int(pt1.y * self.img.shape[0])
                        mask_face.append((x, y))

            face_dict["mask_lip"] = np.array(mask_lip)
            face_dict["mask_face"] = np.array(mask_face)
            face_dict["mask_l_eyes"] = np.array(mask_l_eyes)
            face_dict["mask_r_eyes"] = np.array(mask_r_eyes)
            face_dict["mask_l_eyebrow"] = np.array(mask_l_eyebrow)
            face_dict["mask_r_eyebrow"] = np.array(mask_r_eyebrow)

            full_mask = mask.copy()

            for idx, (part, v) in enumerate(face_dict.items()):
                base = mask.copy()
                convexhull = cv2.convexHull(v)
                if "eyes" in part:
                    color = cfg["eyes"]["color"]
                    weight = cfg["eyes"]["weight"]
                elif "eyebrow" in part:
                    color = cfg["eyebrow"]["color"]
                    weight = cfg["eyebrow"]["weight"]

                elif "face" in part:
                    color = cfg["face"]["color"]
                    weight = cfg["face"]["weight"]

                elif "lip" in part:
                    color = cfg["lip"]["color"]
                    weight = cfg["lip"]["weight"]
                else:
                    color = (0, 0, 0)
                
                
                base = cv2.fillConvexPoly(base, convexhull, (color[2], color[1], color[0]))
                base = cv2.GaussianBlur(base, (7, 7), 20)
                
                full_mask = cv2.addWeighted(full_mask, 1, base, weight, 1)
            tmp = cv2.addWeighted(self.img, 1, full_mask, 1, 1)
            return tmp, full_mask
        return self.img, mask
            

class Camera():
    def __init__(self, cfg, yaml, toml) -> None:
        self.start(**cfg)
        self.config = Config(yaml, toml)
        

    def start(self, index, fps, frame_width, frame_height) -> None:
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        self.mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        

    def load_detector(self, cfg) -> None:
        self.detector = Detector(**cfg)

    # def _load_config(self):
    #     config = Config()


    def capture(self) -> None:
        
        while True:
            success, frame = self.cap.read()
            
            if not success:
                print("SHIT HAPPENED!")
                break

            frame.flags.writeable = False
            
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector(frame)
            
            
            frame.flags.writeable = True
        
            frame, mask = self.detector.post_processing(self.mask, self.config.yaml_cfg)
            
            
            
            
            cv2.imshow("demo", frame)
            cv2.imshow("mask", mask)
            
            
            
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

    CAM = Camera(camera_cfg, yaml_file, toml_file)
    CAM.load_detector(mediapipe_cfg)    
    CAM.capture()