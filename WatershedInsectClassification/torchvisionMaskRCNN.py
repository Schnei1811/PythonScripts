import torch
from torchvision import datasets, models, transforms
import ipdb
from imgaug import augmenters as iaa
from glob import glob
import cv2
import numpy as np
import os
from tqdm import tqdm
import time

batch_size = 1

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X_images, targets, X_paths, img_aug, transform):
        self.X_images = X_images
        self.target = targets
        self.img_aug = img_aug
        self.transform = transform
        self.X_paths = X_paths
    def __len__(self):
        return len(self.X_images)
    def __getitem__(self, idx):
        sample = self.X_images[idx]
        sample = self.img_aug.augment_image(sample).copy()
        sample = sample.astype("float32") / 255.0
        sample = self.transform(sample)
        #print("TRANSFORM", sample)
        return (sample, self.target[idx], self.X_paths[idx])


def yield_fold(X, Y, X_paths, train_val_or_test, chosen_fold=0, kfolds=10):
    #print("TRAIN LABELS", len(Y[kfolds[chosen_fold][TRAIN]]))
    #print("TEST LABELS", len(Y[kfolds[chosen_fold][VAL]]))

    return (X, Y, X_paths)

    # val_index = int(len(X[kfolds[chosen_fold][TRAIN]]) * 0.9)

    # if train_val_or_test == "train":
    #     train_X = X[kfolds[chosen_fold][TRAIN]][:val_index]
    #     train_Y = Y[kfolds[chosen_fold][TRAIN]][:val_index]
    #     train_paths = X_paths[kfolds[chosen_fold][TRAIN]][:val_index]
    #     return (train_X, train_Y, train_paths)
    # elif train_val_or_test == "val":
    #     val_X = X[kfolds[chosen_fold][TRAIN]][:-val_index]
    #     val_Y = Y[kfolds[chosen_fold][TRAIN]][:-val_index]
    #     val_paths = X_paths[kfolds[chosen_fold][TRAIN]][:-val_index]
    #     return (val_X, val_Y, val_paths)
    # else:
    #     test_X = X[kfolds[chosen_fold][TEST]]
    #     test_Y = Y[kfolds[chosen_fold][TEST]]
    #     test_paths = X_paths[kfolds[chosen_fold][TEST]]
    #     return (test_X, test_Y, test_paths)


img_aug = {
    'train': iaa.Sequential([
                iaa.Fliplr(p=0),
            ]),
    'val': iaa.Sequential([
                iaa.Fliplr(p=0)
            ]),
    'test': iaa.Sequential([
                iaa.Fliplr(p=0)
            ])
}

data_transforms ={
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

def buildImage(X_path):
    return cv2.resize(cv2.cvtColor(cv2.imread(X_path), cv2.COLOR_BGR2RGB), (224, 224))


def createData(X_paths, Y, data_name):
    reset = True

    if not os.path.exists("Arrays_Batches"):
        os.makedirs("Arrays_Batches")

    if not os.path.exists("Arrays_Data"):
        os.makedirs("Arrays_Data")

    data_batch = 0
    for i, X_path in enumerate(tqdm(X_paths)):
        if reset == True:
            reset = False
            X = np.expand_dims(buildImage(X_path), axis=0)
        else:
            X = np.vstack((X, np.expand_dims(buildImage(X_path), axis=0)))            
        if not i == 0 and i % 999 == 0:
            reset = True
            np.save(f"Arrays_Batches/{data_name}_Input_{data_batch}_{len(X)}.npy", X)
            np.save(f"Arrays_Batches/{data_name}_Labels_{data_batch}_{len(Y)}.npy", Y)
            data_batch += 1
        if i == len(X_paths) - 1:
            np.save(f"Arrays_Batches/{data_name}_Input_{data_batch}_{len(X)}.npy", X)
            np.save(f"Arrays_Batches/{data_name}_Labels_{data_batch}_{len(Y)}.npy", Y)
            data_batch += 1

    data_paths = []
    for batch in range(data_batch):
        data_paths.append(glob(f'Arrays_Batches/{data_name}_Input_{batch}_*')[0])

    for i, data_path in enumerate(tqdm(data_paths)):
        data = np.load(data_path)
        labels = np.load(data_path.replace("Input", "Labels"))

        if i == 0:
            X = data
        else:
            X = np.vstack((X, data))
        print(X.shape, Y.shape)

    np.save(f'Arrays_Data/{data_name}_Input_{len(X)}.npy', X)
    np.save(f'Arrays_Data/{data_name}_Labels_{len(Y)}.npy', Y)













def test_model(model, 
	dataloaders, 
	device, 
	fold=0, 
	test_img_paths=[], 
	data_name="alus", 
	xp_description="maskrcnn",
	model_name="resnet50", 
	mixed=False, 
	is_inception=False):
    since = time.time()

    class_to_acc = {}
    class_to_count = {}
    preds_array = np.array([])
    labels_array = np.array([])

    global num_to_class
    global class_to_num
    global report_csv

    ipdb.set_trace()

    # Each epoch has a training and validation phase
    for phase in ['test']:
        model.eval()   # Set model to evaluate mode

        running_f1 = 0.0
        running_corrects = 0
        running_corrects_5 = 0
        running_corrects_10 = 0

        # Iterate over data.
        for inputs, labels, paths in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()

            preds_array = np.append(preds_array, preds_cpu)
            labels_array = np.append(labels_array, labels_cpu)

            # forward
            # track history if only in train
            # statistics
            running_f1 += f1_loss(preds, labels.data)
            running_corrects += torch.sum(preds == labels.data)

            outputs_cpu = outputs.detach().cpu().numpy()
            for i in range(outputs_cpu.shape[0]):
                top_10_indices = outputs_cpu[i].argsort()[-10:][::-1]
                for index, k in enumerate(top_10_indices):
                    if int(labels_cpu[i]) == k and index < 5:
                        running_corrects_5 += 1
                        running_corrects_10 += 1
                    elif int(labels_cpu[i]) == k:
                        running_corrects_10 += 1

            #print(preds, labels.data, preds == labels.data)

            #running_corrects_5 += torch.sum(preds == labels.data)

            #print("\nAccuracy", torch.sum(preds == labels.data).double() / batch_size)
            #print("\nModel Prediction", preds_cpu)
            #print("\nLabels", labels_cpu)

            for i, pred in enumerate(preds_cpu):
                label = int(labels_cpu[i])
                img_name = paths[i].split("/")[-1]
                if not num_to_class[label] in class_to_acc:
                    class_to_acc[num_to_class[label]] = 0
                    class_to_count[num_to_class[label]] = 0
                if not img_name in democracy_dict:
                    democracy_dict[img_name] = [pred]
                else:
                    democracy_dict[img_name].append(pred)
                if pred == label:
                    class_to_acc[num_to_class[label]] += 1
                elif mixed.lower() != "true":
                    report_csv.append([img_name, num_to_class[label], num_to_class[pred]])
                else:
                    report_csv.append([img_name, "test_sample", num_to_class[pred]])
                class_to_count[num_to_class[label]] += 1

        epoch_f1 = running_f1 / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        epoch_acc_top5 = running_corrects_5 / len(dataloaders[phase].dataset)
        epoch_acc_top10 = running_corrects_10 / len(dataloaders[phase].dataset)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Confusion Matrix', confusion_matrix(labels_array, preds_array))

    if mixed.lower() != "true":
        print("Class Acc", class_to_acc)

        for i, key in enumerate(class_to_acc):
            print(key,
            class_to_acc[key],
            class_to_count[key],
            round(class_to_acc[key]/class_to_count[key], 3),
            class_to_count[key] - class_to_acc[key])

        print("\n", sum(class_to_acc.values()), " / ", sum(class_to_count.values()),
            '\nAcc: {:4f}'.format(epoch_acc))

        print("\nTop 5 accuracy", epoch_acc_top5)
        print("\nTop 10 accuracy", epoch_acc_top10)

        global test_iter_accuracy
        test_iter_accuracy += (epoch_acc.item() / num_folds)

        print("\n Test Iter Accuracy ", test_iter_accuracy)

        with open(f"Models/{data_name}/{xp_description}/{model_name}/performance_report_{fold}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(report_csv)

        for pred_error in report_csv[1:]:
            dest_dir = f"Models/{data_name}/{xp_description}/{model_name}/pred_error_{fold}/"
            ## Vector


            if "mist" in os.uname()[1]:
                if data_name == "alus":
                    print("ALUS PATH UNIMPLEMENTED")
                elif data_name == "PC":
                    src_path = f"../../alex/ParksCanada/{pred_error[1]}/{pred_error[0]}"
                    file_name = f"{pred_error[1]}_{pred_error[2]}_{pred_error[0]}"
            else:
                print("HOST NAME NOT IMPLEMENTED")
                if data_name == "alus":
                    src_path = f"../../../../../scratch/gobi2/schnei/Alus/ALUS_Full_Data/{pred_error[0]}"
                    file_name = f"{pred_error[0].split('.')[0]}_{pred_error[2]}.jpg"
                else:
                    print("PC image path not implemented")

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            copyfile(src_path, os.path.join(dest_dir, file_name))

        build_confusion_matrix(labels_array, preds_array, [*class_to_acc], fold, epoch_acc, data_name, xp_description, model_name)

    else:
        create_report_csv(data_name, xp_description, model_name, fold)


















if __name__ == "__main__":

	data_name = "alus"
	X_paths = glob("G:PythonData/ALUS/ALUS_Data/cropped_images_div1/*")
	Y = np.zeros(len(X_paths), dtype=int)

	input_file_path = f"Arrays_Data/{data_name}_Input_{len(X_paths)}.npy"
	label_file_path = f"Arrays_Data/{data_name}_Labels_{len(X_paths)}.npy"

	if not os.path.exists(input_file_path):
	    createData(X_paths, Y, data_name)
	X = np.load(input_file_path)
	Y = np.load(label_file_path)

	image_datasets = {x: CustomDataset(*yield_fold(X, Y, X_paths, x), img_aug[x], data_transforms[x]) for x in ['test']}

	dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in ['test']}

	model_ft = models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model_ft = model_ft.to(device)

	ipdb.set_trace()

	test_model(model_ft, dataloaders_dict, device)
