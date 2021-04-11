import urllib.request
import cv2
import numpy as np
import os


def store_raw_images():
    neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01969726'
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()

    pic_num = 1

    if not os.path.exists('octopus'):
        os.makedirs('octopus')

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "octopus/" + str(pic_num) + ".jpg")
            img = cv2.imread("octopus/" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("octopus/" + str(pic_num) + ".jpg", resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))


store_raw_images()