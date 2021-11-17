import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import gui
import door

class Pose_Model:
    def __init__(self, RESNET50_POOLING_AVERAGE, DENSE_LAYER_ACTIVATION, video_path, weight_path = '/Users/joonhyoungjeon/PycharmProjects/pose-estimation/best.hdf5'):
        self.model = Sequential()
        self.model.add(ResNet50(include_top=False, pooling=RESNET50_POOLING_AVERAGE, weights='imagenet'))
        self.model.add(Dense(2, activation=DENSE_LAYER_ACTIVATION))
        self.model.summary()
        self.model.load_weights(weight_path)
        self.video_path = video_path

    def detect_mp_pose(self, image, pose):
        new_image = image.copy()
        image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(new_image)

        landmarks = results.pose_landmarks.landmark
        return landmarks

    def check_in_door(self, point):
        door_min_x = door.min_point[0]
        door_min_y = door.min_point[1]
        door_max_x = door.max_point[0]
        door_max_y = door.max_point[1]

        if door_min_x < point[0] and point[0] < door_max_x:
            if door_min_y < point[1] and point[1] < door_max_y:
                return True
        return False

    def draw_keyPoints_by_landmarks(self, image, landmarks, mp_pose):
        result_image = image.copy()
        image_shape = np.flip(image.shape[:2])
        nose = (landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y)
        R_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
        L_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
        R_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
        L_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
        middle_knee_x = ((landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x + landmarks[
            mp_pose.PoseLandmark.RIGHT_KNEE.value].x) / 2)
        middle_knee_y = ((landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y + landmarks[
            mp_pose.PoseLandmark.RIGHT_KNEE.value].y) / 2)

        cv2.circle(result_image, tuple(np.multiply(nose, image_shape).astype(int)), 10, (0, 0, 255), -1)
        cv2.circle(result_image, tuple(np.multiply(R_shoulder, image_shape).astype(int)), 10, (255, 255, 0),
                   -1)
        cv2.circle(result_image, tuple(np.multiply(L_shoulder, image_shape).astype(int)), 10, (255, 255, 0),
                   -1)
        cv2.circle(result_image, tuple(np.multiply(R_hip, image_shape).astype(int)), 10, (255, 0, 255), -1)
        cv2.circle(result_image, tuple(np.multiply(L_hip, image_shape).astype(int)), 10, (255, 0, 255), -1)
        cv2.circle(result_image, tuple(np.multiply((middle_knee_x, middle_knee_y), image_shape).astype(int)), 10,
                   (255, 0, 0), -1)
        return result_image

    def showProcess(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('FPS:', fps)
        mp_pose = mp.solutions.pose
        sitting_frame=0
        fps_discount = 0
        previous_nose = 0
        disappear_frame = 0
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                if gui.stop:
                    continue
                ret, frame = cap.read()

                if frame is None:
                    break

                if gui.user_sleep is True:
                    if fps_discount < 3:
                        fps_discount = fps_discount + 1
                        continue
                    fps_discount = 0

                image = frame
                try:
                    cv2.rectangle(image, (door.min_point[0], door.min_point[1]),
                                  (door.max_point[0], door.max_point[1])
                                  , (0, 255, 0), 2)

                    landmarks = self.detect_mp_pose(image, pose)

                    image_shape = np.flip(image.shape[:2])
                    nose = (landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y)
                    previous_nose = tuple(np.multiply(nose, image_shape).astype(int))

                    disappear_frame = 0
                    image = self.draw_keyPoints_by_landmarks(image,landmarks, mp_pose)
                    if gui.user_sleep is True:
                        predict_image = np.zeros_like(image)
                        predict_image = self.draw_keyPoints_by_landmarks(predict_image,landmarks, mp_pose)

                        predict_image = cv2.resize(predict_image, (224, 224))
                        predict_image = np.array(predict_image).reshape((1,) + predict_image.shape) * (1. / 255)
                        pred = self.model.predict(predict_image)
                        predicted_class_indices = np.argmax(pred, axis=1)[0]
                        if predicted_class_indices == 0:
                            sitting_frame = 0
                            predict = "Lying"
                        else:
                            sitting_frame = sitting_frame + 1
                            predict = "Sitting"
                            if sitting_frame == 10:
                                sitting_frame = 0
                                gui.user_sleep = False

                        L_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
                        cv2.putText(image, predict,
                                    tuple(np.multiply(L_shoulder, image_shape).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                except Exception as e:
                    disappear_frame = disappear_frame + 1
                    if disappear_frame > 90 and previous_nose != 0:
                        print("disappear")
                        if self.check_in_door(previous_nose):
                            print("In Door")
                        previous_nose = 0
                    pass
                gui.result_img = image
        cap.release()