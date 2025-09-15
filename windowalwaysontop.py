import cv2
import tkinter as tk
from PIL import Image, ImageTk

cap = cv2.VideoCapture(0)

root = tk.Tk()
root.title("Camera Feed")
root.attributes("-topmost", True)  # always on top

lmain = tk.Label(root)
lmain.pack()

def show_frame():
    ret, frame = cap.read()
    if ret:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
    root.after(10, show_frame)

show_frame()
root.mainloop()
cap.release()
