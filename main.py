from PIL import Image, ImageTk
import PIL
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

print(f'A: {A.shape}, u: {u[:,[0]].shape}, vh: {vh[[0],].shape}')

def ith_singular_product(i: int):
    win = tkinter.Tk()
    win.geometry("750x270")
    #Create a canvas
    canvas= tkinter.Canvas(win, width= 600, height= 400)
    canvas.pack()
    #Load an image in the script
    img=ImageTk.PhotoImage(Image.open("actual_grey.png"))
    #Add image to the Canvas Items
    canvas.create_image(10,10,anchor=tkinter.NW,image=img)
    sum = np.zeros(A.shape)
    for count in range(i):
        sum += s[count]*(u[:,[count]] @ vh[[count],])
        pil_image = Image.fromarray(sum)
        img = ImageTk.PhotoImage(pil_image)
        canvas.create_image(10,10,anchor=tkinter.NW,image=img)
        win.update()
        time.sleep(1)

    win.mainloop() 


    return sum

first_singular = ith_singular_product(len(s))

compressed = Image.fromarray(first_singular)
