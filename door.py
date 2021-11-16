import cv2
import numpy as np
class Door:
    def __init__(self):
        self.min_point = [-1, -1]
        self.max_point = [-1, -1]
        self.read_position_file()
        pass

    def read_position_file(self):
        with open("door_position.txt", "r+") as f:
            line = f.readline()
            if line == '':
                self.draw_boundingBox('./testVideo/IMG_5348.MOV')
                f.write('{},{},{},{}'.format(self.min_point[0], self.min_point[1], self.max_point[0], self.max_point[1]))
                return
            points = line.split(',')
            self.min_point[0] = int(points[0])
            self.min_point[1] = int(points[1])
            self.max_point[0] = int(points[2])
            self.max_point[1] = int(points[3])




    def draw_boundingBox(self, video_patb):
        mouse_pressed = False
        rec_mode = "min"
        cap = cv2.VideoCapture(video_patb)
        for _ in range(5):
            ret, frame = cap.read()
        if (frame is None):
            print("Video Error")
            return
        show_img = frame
        img = np.copy(show_img)
        def mouse_callback(event, _x, _y, flags, param):
            nonlocal show_img, mouse_pressed
            if event == cv2.EVENT_LBUTTONDOWN:
                print("Mouse Press!")
                if rec_mode is "min":
                    self.min_point[0] = _x
                    self.min_point[1] = _y
                if rec_mode is "max":
                    self.max_point[0] = _x
                    self.max_point[1] = _y
                mouse_pressed = True
                show_img = np.copy(img)
                if self.min_point[0] != -1:
                    cv2.circle(show_img, (self.min_point[0], self.min_point[1]),
                                  8, (255, 0, 0), -1)
                if self.max_point[0] != -1:
                    cv2.circle(show_img, (self.max_point[0], self.max_point[1]),
                               8, (0, 0, 255), -1)

            elif event == cv2.EVENT_LBUTTONUP:
                print("Mouse Down!")
                mouse_pressed = False
                if self.min_point[0] != -1 and self.max_point[0] != -1:
                    cv2.putText(show_img, "Min",
                                (self.min_point[0], self.min_point[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA
                                )
                    cv2.putText(show_img, "Max",
                                (self.max_point[0], self.max_point[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA
                                )
                    cv2.rectangle(show_img, (self.min_point[0], self.min_point[1]),(self.max_point[0], self.max_point[1])
                                  ,(0, 255, 0), 2)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', mouse_callback)

        while True:
            cv2.imshow('image', show_img)
            k = cv2.waitKey(1)

            if k == ord('a') and not mouse_pressed:
                print("종료")
                break
            if k == ord('s') and not mouse_pressed:
                if rec_mode is 'min':
                    rec_mode = 'max'
                else:
                    rec_mode = 'min'