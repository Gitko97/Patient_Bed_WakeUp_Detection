import threading
import tkinter
import PIL
import cv2
from PIL import ImageTk as Pil_imageTk
result_img = 0
user_sleep = False
class App(threading.Thread):
    def __init__(self, window, window_title):
        threading.Thread.__init__(self)
        self.window = window
        self.result_img = 0
        self.window.title(window_title)
        self.window.resizable(False, False)
        self.delay = 1
        # View Video
        self.video_width = 1080
        self.video_height = 920
        self.frame1 = tkinter.Frame(window, relief="solid", bd=2)
        self.frame1.pack(side="left", fill="both", expand=True)

        self.frame2 = tkinter.Frame(window, relief="solid", bd=2)
        self.frame2.pack(side="right", fill="both", expand=True)

        self.canvas = tkinter.Canvas(self.frame1, width=self.video_width, height=self.video_height)
        self.canvas.pack()
        self.button1 = tkinter.Button(self.frame2, text="문 설정", overrelief='solid', height=15,command=None,compound="c")
        self.button1.pack(expand=1)
        self.button2 = tkinter.Button(self.frame2, text="환자 취침", overrelief='solid', height=15,command=self.change_sleep,compound="c", fg='green')
        self.button2.pack(expand=1)
        self.update()
        self.window.mainloop()

    def change_sleep(self):
        global user_sleep
        user_sleep = True
        self.button2.configure(fg="red")

    def update(self):
        global result_img
        try:
            if user_sleep is False:
                self.button2.configure(fg="green")
            vid = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            vid = cv2.resize(vid, (self.video_width, self.video_height))
            self.photo = Pil_imageTk.PhotoImage(image=PIL.Image.fromarray(vid))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        except Exception as e:
            pass
        self.window.after(self.delay, self.update)
