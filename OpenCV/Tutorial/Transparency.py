import numpy as np
import cv2

image = cv2.imread("Files/thumbsup.jpg")

for alpha in np.arange(0, 1.1, 0.1)[::-1]:
    overlay = image.copy()
    output = image.copy()

    cv2.rectangle(overlay, (20, 20), (80, 80), (0, 0, 255), -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    print("alpha={}, beta={}".format(alpha, 1 - alpha))
    cv2.imshow("Output", output)
    cv2.waitKey(0)