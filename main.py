from PIL import Image, ImageTk
from numpy import asarray
from numpy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import tkinter




image = Image.open("actual_grey.png")
#image.show()

A = asarray(image)

u, s, vh = svd(A)

print(f'A: {A.shape}, u: {u.shape}, vh: {vh.shape}')

def ith_singular_product(i: int):
    win = tkinter.Tk()
    win.geometry("750x270")
    #Create a canvas
    canvas= tkinter.Canvas(win, width= 600, height= 400)
    canvas.pack()
    text_str = tkinter.StringVar()
    text = tkinter.Label(win, textvariable=text_str)
    text.pack()
    #Load an image in the script
    sum = np.zeros(A.shape)
    for count in range(i):
        sum += s[count]*(u[:,[count]] @ vh[[count],])
        pil_image = Image.fromarray(sum)
        img = ImageTk.PhotoImage(pil_image)
        canvas.create_image(10,10,anchor=tkinter.NW,image=img)
        percent = count * (u.shape[0] + vh.shape[0]) / (A.shape[0] * A.shape[1])*100
        text_str.set(f'Singular Values: {count}\n{percent:.2f}% of original size')
        win.update()
        time.sleep(0.2)

    win.mainloop() 


    return sum

first_singular = ith_singular_product(len(s))

compressed = Image.fromarray(first_singular)
