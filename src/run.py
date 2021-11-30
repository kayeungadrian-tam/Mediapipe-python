import cv2
import yaml
import sys
from loguru import logger
import numpy as np
import mediapipe as mp

class Config():
    def __init__(self, yaml_file) -> None:
        logger.info(f"Reading config file: {yaml_file}")
        with open(yaml_file, "r") as f:
            self.yaml_cfg = yaml.safe_load(f)

class Detector():
    def __init__(self, 
                thickness=1, 
                circle_radius=1, 
                color=(255,0,255), 
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5) -> None:
        logger.info("Mediapipe detector initiated.")
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
        # return self.results

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

            for part, v in face_dict.items():
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
                
                full_mask = cv2.addWeighted(full_mask, 1, base, weight, 1)
            full_mask = cv2.GaussianBlur(full_mask, (7, 7), 20)
            tmp = cv2.addWeighted(self.img, 1, full_mask, 1, 1)
            
            return tmp, full_mask
        logger.warning("Face not detected.")
        return self.img, mask
            

class Camera():
    def __init__(self, index, config, detector) -> None:
        self.start(index)
        self.config = config
        self.detector = detector

    def start(self, index) -> None:
        self.cap = cv2.VideoCapture(index)
        success, frame = self.cap.read()
        if not success:
            logger.error(f"Camera not successful: video input: {index}")
            sys.exit()
        self.mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        
    def capture(self) -> None:
        logger.info("Catpuring images from video input...")
        while True:
            
            _, frame = self.cap.read()
            
            frame.flags.writeable = False
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.detector(frame)
            
            frame.flags.writeable = True
            frame, mask = self.detector.post_processing(self.mask, self.config.yaml_cfg)
            
            frame = cv2.flip(frame, 1)
            
            cv2.imshow("demo", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Process terminated.")


def main(yaml_file):
    config = Config(yaml_file)
    detector = Detector()
    CAM = Camera(0, config, detector)
    CAM.capture()

if __name__ == "__main__":
    
    yaml_file = "config/makeup.yaml"
    
    main(yaml_file)