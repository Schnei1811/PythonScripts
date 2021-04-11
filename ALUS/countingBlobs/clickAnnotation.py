from pynput import mouse
from glob import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import png
import os

global input_lst
input_lst = []


#img_dir = "C:\\Users\\Stefan\\Desktop\\CountBlobs\\LCFCN-master\\ALUS_F\\images"
#img_dir = "C:\\Users\\Stefan\\Desktop\\CountBlobs\\LCFCN-master\\ALUS_BLF\\images"
# img_dir = "C:\\Users\\Stefan\\Desktop\\CountBlobs\\LCFCN-master\\ALUS_BL\\images"
img_dir = "C:\\Users\\Stefan\\Desktop\\CountBlobs\\LCFCN-master\\ALUS_BL\\full_data"

sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for size in sizes:
    if not os.path.exists(img_dir + "_div{}".format(size)):
        os.makedirs(img_dir + "_div{}".format(size))

# def on_move(x, y):
#     print ("Mouse moved to ({0}, {1})".format(x, y))

def on_click(x, y, button, pressed):
    y = y-35
    print(button)
    global input_lst, img, annotating
    if pressed:
        if str(button) == "Button.right":
            if len(input_lst) > 0:
                x, y = input_lst[-1][0], input_lst[-1][1]
                img[y - 3:y + 3, x - 3:x + 3] = [255, 255, 255]
                input_lst.pop()
                print(input_lst)
        if str(button) == "Button.x1":
            coord = [x, y]
            print('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
            input_lst.append(coord)
            print(input_lst)
            img[y-3:y+3, x-3:x+3] = [0, 0, 255]
        elif str(button) == "Button.x2":
            annotating = False
            for i in sizes:
                save_img_sizes(i)
            input_lst = []

# def on_scroll(x, y, dx, dy):
#     print('Mouse scrolled at ({0}, {1})({2}, {3})'.format(x, y, dx, dy))

# with Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
#     listener.join()


def save_img_sizes(div):
    global h, w, c
    print(img_dir)

    div_path = img_dir + "_div{}\\".format(div)

    img_name = div_path + img_path.split("\\")[-1]

    resized_img = cv2.resize(cv2.imread(img_path), (int(w / div), int(h / div)))
    cv2.imwrite(img_name, resized_img)
    resize_h, resize_w, resize_c = resized_img.shape
    data = np.zeros((resize_h, resize_w, 3), dtype=np.uint8)
    for points in input_lst:
        data[int(points[1]/ div), int(points[0]/div)] = [255, 0, 0]
    img_dots = Image.fromarray(data, "RGB")
    img_dots.save(div_path + img_name_dots)


    with open(div_path + img_name_text, "w") as f:
        for i, list in enumerate(input_lst):
            if i != 0:
                f.write("\n")
            for k, string in enumerate(list):
                if k != 0:
                    f.write("\t")
                f.write(str(int(string / div)))
    print("Right", img_name)


    cv2.destroyAllWindows()

    pass


listener = mouse.Listener(on_click=on_click)
listener.start()

for i, img_path in enumerate(glob(img_dir + "\\*")):
    print(img_path)
    global img_name, annotating
    img = cv2.imread(img_path)
    annotating = True
    if ".txt" in img_path or "dots" in img_path:
        pass
    else:
        while annotating:
            img_name_dots = img_path.split("\\")[-1][:-4] + 'dots.png'
            img_name_text = img_path.split("\\")[-1][:-4] + ".txt"
            global h, w, c
            h, w, c = img.shape
            cv2.imshow('image', img)
            cv2.waitKey(5)