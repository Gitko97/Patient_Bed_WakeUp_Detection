import cv2
import numpy as np
door_min_point = [-1, -1]
door_max_point = [-1, -1]
pillow_min_point = [-1, -1]
pillow_max_point = [-1, -1]
class Door:
    def __init__(self):
        pass

    def read_position_file(self, set_door, video_path):
        global door_min_point, door_max_point, pillow_min_point, pillow_max_point
        points = []
        with open("position.txt", "r+") as f:
            for _ in range(2):
                line = f.readline()
                if line == '' or set_door:
                    print("Door Not detected!")
                    self.draw_boundingBox(video_path)
                    if door_min_point[0] == -1 or door_max_point[0] == -1 or pillow_min_point[0] == -1 or pillow_max_point[0] == -1:
                        print("Please set door bounding box!")
                        return True
                    with open("position.txt", "w") as writeFile:
                        writeFile.write('{},{},{},{}\n'.format(door_min_point[0], door_min_point[1], door_max_point[0], door_max_point[1]))
                        writeFile.write('{},{},{},{}'.format(pillow_min_point[0], pillow_min_point[1], pillow_max_point[0], pillow_max_point[1]))
                    return True
                point = line.split(',')
                points.append(point)

        door_min_point[0] = int(points[0][0])
        door_min_point[1] = int(points[0][1])
        door_max_point[0] = int(points[0][2])
        door_max_point[1] = int(points[0][3])

        pillow_min_point[0] = int(points[1][0])
        pillow_min_point[1] = int(points[1][1])
        pillow_max_point[0] = int(points[1][2])
        pillow_max_point[1] = int(points[1][3])

        return False




    def draw_boundingBox(self, video_patb):
        mouse_pressed = False
        DRAW_MODE_LIST = {"Door": 0, "Pillow" : 1}
        REC_MODE_LIST = {"Min": 0, "Max" : 1}

        current_recMode = REC_MODE_LIST["Min"]
        current_drawMode = DRAW_MODE_LIST["Door"]

        cap = cv2.VideoCapture(video_patb)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 1200)

        for _ in range(5):
            ret, frame = cap.read()
        if (frame is None):
            print("Video Error")
            return
        show_img = frame
        original_img = np.copy(show_img)

        def mouse_callback(event, _x, _y, flags, param):
            nonlocal show_img,current_recMode,current_drawMode, mouse_pressed, original_img
            global door_min_point, door_max_point, pillow_min_point, pillow_max_point
            if event == cv2.EVENT_LBUTTONDOWN:
                if current_recMode is REC_MODE_LIST["Min"]:
                    if current_drawMode is DRAW_MODE_LIST["Door"]:
                        door_min_point[0] = _x
                        door_min_point[1] = _y
                    if current_drawMode is DRAW_MODE_LIST["Pillow"]:
                        pillow_min_point[0] = _x
                        pillow_min_point[1] = _y
                if current_recMode is REC_MODE_LIST["Max"]:
                    if current_drawMode is DRAW_MODE_LIST["Door"]:
                        door_max_point[0] = _x
                        door_max_point[1] = _y
                    if current_drawMode is DRAW_MODE_LIST["Pillow"]:
                        pillow_max_point[0] = _x
                        pillow_max_point[1] = _y
                mouse_pressed = True
                show_img = np.copy(original_img)
                if door_min_point[0] != -1:
                    cv2.circle(show_img, (door_min_point[0], door_min_point[1]),
                               8, (255, 0, 0), -1)
                if door_max_point[0] != -1:
                    cv2.circle(show_img, (door_max_point[0], door_max_point[1]),
                               8, (0, 0, 255), -1)
                if pillow_min_point[0] != -1:
                    cv2.circle(show_img, (pillow_min_point[0], pillow_min_point[1]),
                               8, (255, 0, 0), -1)
                if pillow_max_point[0] != -1:
                    cv2.circle(show_img, (pillow_max_point[0], pillow_max_point[1]),
                               8, (0, 0, 255), -1)

            elif event == cv2.EVENT_LBUTTONUP:
                mouse_pressed = False
                if door_min_point[0] != -1 and door_max_point[0] != -1:
                    cv2.putText(show_img, "Door",
                                (door_min_point[0], door_min_point[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA
                                )
                    cv2.rectangle(show_img, (door_min_point[0], door_min_point[1]), (door_max_point[0], door_max_point[1])
                                  , (0, 255, 0), 2)
                if pillow_min_point[0] != -1 and pillow_max_point[0] != -1:
                    cv2.putText(show_img, "Pillow",
                                (pillow_min_point[0], pillow_min_point[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA
                                )
                    cv2.rectangle(show_img, (pillow_min_point[0], pillow_min_point[1]), (pillow_max_point[0], pillow_max_point[1])
                                  , (0, 255, 0), 2)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', mouse_callback)
        height, width, channel = show_img.shape
        while True:
            k = cv2.waitKey(1)

            if k == ord('a') and not mouse_pressed:
                print("종료")
                break
            if k == ord('s') and not mouse_pressed:
                if current_recMode is REC_MODE_LIST["Max"]:
                    current_recMode = REC_MODE_LIST["Min"]
                else:
                    current_recMode = REC_MODE_LIST["Max"]

            if k == ord('m') and not mouse_pressed:
                if current_drawMode is DRAW_MODE_LIST["Door"]:
                    current_drawMode = DRAW_MODE_LIST["Pillow"]
                else:
                    current_drawMode = DRAW_MODE_LIST["Door"]

            cv2.putText(show_img, "Current : "+str(list(DRAW_MODE_LIST.keys())[current_drawMode]) +" / "+ str(list(REC_MODE_LIST.keys())[current_recMode]),
                        (0, height - 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            cv2.putText(show_img, "Press 'A' to close", (0, height- 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            cv2.putText(show_img, "'S' to change Min/Max point",
                        (0, height - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            cv2.putText(show_img, "'M' to change Mode",
                        (0, height),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

            cv2.imshow('image', show_img)

        cv2.destroyAllWindows()