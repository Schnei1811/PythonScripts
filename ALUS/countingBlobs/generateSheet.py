import cv2
from glob import glob
import random
import sys
import numpy as np
from tqdm import tqdm
import pickle

empty_sheet_path = "G:\\PythonData\ALUS\\21_03_13_Sheet\\empty_sheet.jpg"



sheet = cv2.imread(empty_sheet_path)


extracted_imgs_paths = glob("G:\\PythonData\\ALUS\\21_03_13_Alus_Dish_Sheet_Data\\alus_sheet\\*")

sheet = cv2.resize(sheet, (int(sheet.shape[1]*1.25), int(sheet.shape[0]*1.25)))
# sheet = cv2.resize(sheet, (int(sheet.shape[1]*0.25), int(sheet.shape[0]*0.25)))

sheet_h = sheet.shape[0]
sheet_w = sheet.shape[1]

ex = 0

bug_dict = {}

for ex in tqdm(range(1050)):
    try:
        random.shuffle(extracted_imgs_paths)
        img_path = extracted_imgs_paths.pop()

        img = cv2.imread(img_path)
        alpha_image = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        for i, x in enumerate(img):
            for k, y in enumerate(x):
                if img[i][k][0] > 200 and img[i][k][1] > 200 and img[i][k][2] > 200:
                    alpha_image[i][k][3] = 0

        counter = 0
        while True:
            exit = True
            x1 = random.randrange(300, sheet_w - 300 - img.shape[1])
            x2 = x1 + alpha_image.shape[1]
            y1 = random.randrange(300, sheet_h - 300 - img.shape[0])
            y2 = y1 + alpha_image.shape[0]

            bw_img = cv2.cvtColor(sheet[y1-10: y2+10, x1-10: x2+10], cv2.COLOR_BGR2GRAY)
            t = bw_img.flatten().tolist()
            for val in t:
                if val < 225:
                    exit = False
                    counter += 1
                    print("overlap")
                    break
            if exit or counter == 2000:
                break

        if counter == 2000:
            continue

        bkgrnd_img = sheet[y1: y2, x1: x2, :]

        alpha_channel = alpha_image[:, :, 3]
        rgb_channels = alpha_image[:, :, :3]

        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

        base = rgb_channels.astype(np.float32) * alpha_factor
        white = bkgrnd_img.astype(np.float32) * (1 - alpha_factor)
        final_image = (base + white).astype(np.uint8)

        canvas = bkgrnd_img
        canvas[0:final_image.shape[0] + 0, 0:final_image.shape[1] + 0, :] = final_image

        # cv2.imshow("img", cv2.resize(sheet, (1200, 1600)))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        cv2.imwrite(f"images/test_{ex}.jpg", sheet)

        order = "_".join(img_path.split("\\")[-1].split("_")[:2])

        if not order in bug_dict:
            bug_dict[order] = 1
        else:
            bug_dict[order] += 1

        print("i", ex)
        # print(bug_dict, img_path)

        with open('sheet_bug_dict.pkl', 'wb') as handle:
            pickle.dump(bug_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except:
        print("error", sys.exc_info())

# import ipdb;
# ipdb.set_trace()


# Araneae_Unknown 57                61
# Coleoptera_Unknown 71             72
# Collembola_Unknown 41             40
# Dermaptera_Unknown 27             28
# Diptera_Unknown 73                71
# EPT_Unknown 13                    12
# Hemiptera_Unknown 75              76
# Hymenoptera_Apoidea 61            75
# Hymenoptera_Formicidae 56         58
# Hymenoptera_Unknown 65            58
# Larvae_Unknown 65                 76
# Lepidoptera_Unknown 79            78
# Mites_Unknown 19                  18
# Neuroptera_Unknown 39             34
# Non-miteArachnids_Unknown 76      71
# Orthoptera_Unknown 43             44
# Plecoptera_Unknown 36             36
# Psocodea_Unknown 81               72
# Thysanoptera_Unknown 45           64