import threading  # Thread
import tkinter

from controller import Controller
from gui import App
from model import Pose_Model
from door import Door
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
weight_path = './best.hdf5'

result_img = 0
if __name__ == "__main__":
    pose_model = Pose_Model(RESNET50_POOLING_AVERAGE, DENSE_LAYER_ACTIVATION, weight_path=weight_path)
    t1 = threading.Thread(target=pose_model.showProcess, args=())
    t1.daemon = True
    t1.start()
    app = App(tkinter.Tk(), "GUI")
    app.daemon = True
    app.start()
    # Door()