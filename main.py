import argparse
import threading  # Thread
import tkinter

from gui import App
from model import Pose_Model
from door import Door
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
weight_path = './best.hdf5'

result_img = 0
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--door', '-d', help='Setting new door', default=False, dest='door', type=str2bool)
    parser.add_argument('--phone', '-p', help='Setting phone number', default=False, dest='phone', type=str2bool)
    parser.add_argument('--weight', '-w', help='Setting weight path', default="./best.hdf5", dest='weight_path')
    parser.add_argument('--video', '-v', help='Input video path', dest='video')

    set_door = parser.parse_args().door
    set_phone = parser.parse_args().phone
    weight_path = parser.parse_args().weight_path
    video_path = parser.parse_args().video

    video_path = './testVideo/test.mov' # debug path

    door = Door()
    set_door = door.read_position_file(set_door, video_path)

    if not set_door and not set_phone:
        pose_model = Pose_Model(RESNET50_POOLING_AVERAGE, DENSE_LAYER_ACTIVATION,video_path, weight_path=weight_path)
        t1 = threading.Thread(target=pose_model.showProcess, args=())
        t1.daemon = True
        t1.start()
        app = App(tkinter.Tk(), "GUI")
        app.daemon = True
        app.start()