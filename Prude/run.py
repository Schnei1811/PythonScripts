from nsfw_detector import predict
import os
import cv2
import pickle

yahoo_pickle = "yahoo_results.pkl"

# import tensorflow.compat.v1 as tf
#
# from yahoo_image_utils import create_tensorflow_image_loader
# from yahoo_image_utils import create_yahoo_image_loader
# from yahoo_model import OpenNsfwModel, InputType
#
# import numpy as np
# import base64
#
# IMAGE_LOADER_TENSORFLOW = "tensorflow"
# IMAGE_LOADER_YAHOO = "yahoo"
#
# yahoo_model = OpenNsfwModel()
#
# result_yahoo = {}
#
# path = "C:\\Users\\Stefan\\Desktop\\PythonScripts\\Prude\\test_images\\"
#
# with tf.Session() as sess:
#
#     yahoo_model.build(weights_path="yahoo_open_nsfw-weights.npy", input_type=InputType.BASE64_JPEG)
#
#     for file in os.listdir("test_images"):
#         fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])
#         sess.run(tf.global_variables_initializer())
#         image = fn_load_image("test_images\\" + file)
#         predictions = sess.run(yahoo_model.predictions, feed_dict={yahoo_model.input: image})
#         print("Results for '{}'".format(path + file))
#         print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))
#         # import ipdb;ipdb.set_trace()
#         result_yahoo[path + file] = {}
#         result_yahoo[path + file]["not porn"] = predictions[0][0]
#         result_yahoo[path + file]["porn"] = predictions[0][1]
#
# sess.close()
#
# with open(yahoo_pickle, "wb") as f:
#     pickle.dump(result_yahoo, f, pickle.HIGHEST_PROTOCOL)

with open(yahoo_pickle, "rb") as f:
    result_yahoo = pickle.load(f)

model_name_inception = './nsfw.299x299.h5'
model_inception_net = predict.load_model(model_name_inception)
result_inception = predict.classify(model_inception_net, 'test_images/', 299)

model_name_mobile = './nsfw_mobilenet2.224x224.h5'
model_mobile_net = predict.load_model(model_name_mobile)
result_mobile = predict.classify(model_mobile_net, 'test_images/', 244)


acc_yahoo = 0
acc_inception = 0
acc_mobile = 0
acc_ensemble = 0

result_ensemble = {}

for i, path in enumerate(result_mobile):
    print("\n", path)
    img_name = path.split('\\')[-1]
    porn = False
    porn_votes = 0
    if "nsfw" in img_name:
        porn = True
    cv2_img = cv2.imread(path)
    cv2.imshow(path.split('\\')[-1], cv2.resize(cv2_img, (cv2_img.shape[1] * 2, cv2_img.shape[0] * 2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    yahoo_pred_class = max(result_yahoo[path], key=result_yahoo[path].get)
    if yahoo_pred_class == "porn":
        porn_votes += 1
    if yahoo_pred_class == "porn" and porn == True or yahoo_pred_class != "porn" and porn == False:
        acc_yahoo += 1
    print("\033[1m", "\nYahooNet: ", '\033[0m', yahoo_pred_class, " Correct:  ", acc_yahoo, " Total: ", i + 1, " Accuracy: \033[1m",
          acc_yahoo / (i + 1), '\033[0m')
    for classification in result_yahoo[path]:
        if classification == yahoo_pred_class:
            print('\033[1m', classification, result_yahoo[path][classification], '\033[0m')
        else:
            print(classification, result_yahoo[path][classification])

    mobile_pred_class = max(result_mobile[path], key=result_mobile[path].get)
    if mobile_pred_class == "porn":
        porn_votes += 1
    if mobile_pred_class == "porn" and porn == True or mobile_pred_class != "porn" and porn == False:
        acc_mobile += 1
    print("\033[1m", "\nMobileNet: ", '\033[0m', mobile_pred_class, " Correct:  ", acc_mobile, " Total: ", i + 1, " Accuracy: \033[1m", acc_mobile / (i + 1), '\033[0m')
    for classification in result_mobile[path]:
        if classification == mobile_pred_class:
            print('\033[1m', classification, result_mobile[path][classification], '\033[0m')
        else:
            print(classification, result_mobile[path][classification])


    inception_pred_class = max(result_inception[path], key=result_inception[path].get)
    if inception_pred_class == "porn":
        porn_votes += 1
    if inception_pred_class == "porn" and porn == True or inception_pred_class != "porn" and porn == False:
        acc_inception += 1
    print("\033[1m", "\nInceptionNet: ", '\033[0m', inception_pred_class, " Correct: ", acc_inception, " Total: ", i + 1, " Accuracy: \033[1m", acc_inception / (i + 1), '\033[0m')
    for classification in result_inception[path]:
        if classification == inception_pred_class:
            print('\033[1m', classification, result_inception[path][classification], '\033[0m')
        else:
            print(classification, result_inception[path][classification])

    result_ensemble[path] = {}
    result_ensemble[path]["not porn"] = (3 - porn_votes) / 3
    result_ensemble[path]["porn"] = porn_votes / 3

    ensemble_pred_class = "porn"
    if porn_votes < 2:
        ensemble_pred_class = "not porn"

    if porn_votes >= 2 and porn == True:
        acc_ensemble += 1
    elif porn_votes < 2 and porn == False:
        acc_ensemble += 1

    print("\033[1m", "\nEnsemble: ", '\033[0m', ensemble_pred_class, " Correct: ", acc_ensemble, " Total: ", i + 1, "\033[1m Accuracy: ", acc_ensemble / (i + 1), '\033[0m')
    for classification in result_ensemble[path]:
        if classification == ensemble_pred_class:
            print('\033[1m', classification, result_ensemble[path][classification], '\033[0m')
        else:
            print(classification, result_ensemble[path][classification])

    cv2.imshow(path.split('\\')[-1], cv2.resize(cv2_img, (cv2_img.shape[1] * 2, cv2_img.shape[0] * 2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print("\nAccuracy YahooNet", acc_yahoo, len(os.listdir("test_images")), acc_yahoo / len(os.listdir("test_images")))
print("\nAccuracy MobileNet", acc_mobile, len(os.listdir("test_images")), acc_mobile / len(os.listdir("test_images")))
print("\nAccuracy InceptionNet", acc_inception, len(os.listdir("test_images")), acc_inception / len(os.listdir("test_images")))
print("\nAccuracy Ensemble", acc_ensemble, len(os.listdir("test_images")), acc_ensemble / len(os.listdir("test_images")))