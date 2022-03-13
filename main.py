from PIL import Image, ImageTk
from numpy import asarray
from numpy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from sys import argv
import tkinter


def main(filename):

    image = Image.open(filename)

    A = asarray(image)

    u, s, vh = svd(A)

    win = tkinter.Tk()
    win.geometry("750x270")
    canvas= tkinter.Canvas(win, width= 600, height= 400)
    canvas.pack()
    text_str = tkinter.StringVar()
    text = tkinter.Label(win, textvariable=text_str)
    text.pack()
    sum = np.zeros(A.shape)
    for i in range(len(s)):
        sum += s[i]*(u[:,[i]] @ vh[[i],])
        pil_image = Image.fromarray(sum)
        img = ImageTk.PhotoImage(pil_image)
        canvas.create_image(10,10,anchor=tkinter.NW,image=img)
        percent = i * (u.shape[0] + vh.shape[0]) / (A.shape[0] * A.shape[1])*100
        text_str.set(f'Singular Values: {i} of {len(s)}\n{percent:.2f}% of original size\nCurrent SV: {s[i]:.3e}\nMax SV: {s[0]:e}')
        win.update()
        time.sleep(0.2)

    win.mainloop() 


if __name__ == '__main__':
    main(argv[1])
