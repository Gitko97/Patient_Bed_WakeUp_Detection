gui_thread = 0
model_thread = 0
class Controller:
    def __init__(self, gui, model):
        global gui_thread, model_thread
        gui_thread = gui
        model_thread = model

    def stop_thread(self):
        global gui_thread, model_thread
        print(gui_thread.is_alive())
        print(model_thread.is_alive())
        return