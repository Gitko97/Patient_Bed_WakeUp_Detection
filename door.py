import cv2
import numpy as np
min_point = [-1, -1]
max_point = [-1, -1]
class Door:
    def __init__(self):
        pass

    def read_position_file(self, set_door, video_path):
        global min_point, max_point
        with open("door_position.txt", "r+") as f:
            line = f.readline()
            if line == '' or set_door:
                print("Door Not detected!")
                self.draw_boundingBox(video_path)
                if min_point == [-1, -1] or max_point == [-1, -1]:
                    print("Please set door bounding box!")
                    return True
                f.write('{},{},{},{}'.format(min_point[0], min_point[1], max_point[0], max_point[1]))
                return True
            points = line.split(',')
            min_point[0] = int(points[0])
            min_point[1] = int(points[1])
            max_point[0] = int(points[2])
            max_point[1] = int(points[3])
            return False




    def draw_boundingBox(self, video_patb):
        mouse_pressed = False
        rec_mode = "min"
        cap = cv2.VideoCapture(video_patb)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 750)
        for _ in range(5):
            ret, frame = cap.read()
        if (frame is None):
            print("Video Error")
            return
        show_img = frame
        img = np.copy(show_img)

        def mouse_callback(event, _x, _y, flags, param):
            global  min_point, max_point
            nonlocal show_img, mouse_pressed
            if event == cv2.EVENT_LBUTTONDOWN:
                print("Mouse Press!")
                if rec_mode is "min":
                    min_point[0] = _x
                    min_point[1] = _y
                if rec_mode is "max":
                    max_point[0] = _x
                    max_point[1] = _y
                mouse_pressed = True
                show_img = np.copy(img)
                if min_point[0] != -1:
                    cv2.circle(show_img, (min_point[0], min_point[1]),
                                  8, (255, 0, 0), -1)
                if max_point[0] != -1:
                    cv2.circle(show_img, (max_point[0], max_point[1]),
                               8, (0, 0, 255), -1)

            elif event == cv2.EVENT_LBUTTONUP:
                print("Mouse Down!")
                mouse_pressed = False
                if min_point[0] != -1 and max_point[0] != -1:
                    cv2.putText(show_img, "Min",
                                (min_point[0], min_point[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA
                                )
                    cv2.putText(show_img, "Max",
                                (max_point[0], max_point[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA
                                )
                    cv2.rectangle(show_img, (min_point[0], min_point[1]),(max_point[0], max_point[1])
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