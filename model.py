import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import gui
from Managing_dementia_patients.door import Door


class Pose_Model:
    def __init__(self, RESNET50_POOLING_AVERAGE, DENSE_LAYER_ACTIVATION, weight_path = '/Users/joonhyoungjeon/PycharmProjects/pose-estimation/best.hdf5'):
        self.model = Sequential()
        self.model.add(ResNet50(include_top=False, pooling=RESNET50_POOLING_AVERAGE, weights='imagenet'))
        self.model.add(Dense(2, activation=DENSE_LAYER_ACTIVATION))
        self.model.summary()
        self.model.load_weights(weight_path)
        # self.door = Door()

    def detect_mp_pose(self, image, pose):
        new_image = image.copy()
        image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(new_image)

        landmarks = results.pose_landmarks.landmark
        return landmarks

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

    def showProcess(self, video_path='./testVideo/IMG_5348.MOV'):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('FPS:', fps)
        mp_pose = mp.solutions.pose
        sitting_frame=0
        fps_discount = 0
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                if frame is None:
                    break

                if fps_discount < 4:
                    fps_discount = fps_discount + 1
                    continue
                fps_discount = 0

                image = frame
                try:
                    # cv2.rectangle(image, (self.door.min_point[0], self.door.min_point[1]),
                    #               (self.door.max_point[0], self.door.max_point[1])
                    #               , (0, 255, 0), 2)
                    # if gui.user_sleep is False:
                    landmarks = self.detect_mp_pose(image, pose)

                    image = self.draw_keyPoints_by_landmarks(image,landmarks, mp_pose)
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
                    image_shape = np.flip(image.shape[:2])
                    L_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
                    cv2.putText(image, predict,
                                tuple(np.multiply(L_shoulder, image_shape).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 255), 2, cv2.LINE_AA
                                )

                except Exception as e:
                    print(e)
                    pass
                gui.result_img = image
        cap.release()