import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import power_transform
from scipy import stats
import copy

order_dict = {
    "araneae": 0,
    "coleoptera": 0,
    "collembola": 0,
    "dermaptera": 0,
    "diptera": 0,
    "hemiptera": 0,
    "hymenoptera_apoidea": 0,
    "hymenoptera_formicidae": 0,
    "hymenoptera_unknown": 0,
    "larvae": 0,
    "lepidoptera": 0,
    "mites": 0,
    "neuroptera": 0,
    "non-miteArachnid": 0,
    "orthoptera": 0,
    "plecoptera": 0,
    "psocodea": 0,
    "thysanoptera": 0
}



data = {
    "araneae":
        {
            "train":
                {
                    "data": np.array([111169, 286627, 42634, 40012]),
                    "label": np.array([0.4761, 0.0527, 0.0886, 0.1063])
                },
            "test":
                {
                    "data": np.array([95309]),
                    "label": np.array([0.3535])
                },
        },
    "coleoptera":
        {
            "train":
                {
                    "data": np.array([75099, 108119, 109516, 112707]),
                    "label": np.array([0.1312, 0.3212, 0.3366, 0.1842])
                },
            "test":
                {
                    "data": np.array([94135]),
                    "label": np.array([0.2709])
                },
        },
    "collembola":
        {
            "train":
                {
                    "data": np.array([3442, 5752, 1128, 3407]),
                    "label": np.array([0.0002515, 0.00079, 0.0001448, 0.0018188])
                },
            "test":
                {
                    "data": np.array([677]),
                    "label": np.array([0.0001448])
                },
        },
    "dermaptera":
        {
            "train":
                {
                    "data": np.array([219649]),
                    "label": np.array([0.6917])
                },
            "test":
                {
                    "data": np.array([292641]),
                    "label": np.array([0.7667])
                },
        },
    "diptera":
        {
            "train":
                {
                    "data": np.array([81247, 66806, 28193, 153305]),
                    "label": np.array([0.1464, 0.1608, 0.0857, 0.5227])
                },
            "test":
                {
                    "data": np.array([45774]),
                    "label": np.array([0.1203])
                },
        },
    "hemiptera":
        {
            "train":
                {
                    "data": np.array([41525, 52713, 15478, 27955]),
                    "label": np.array([0.0916, 0.1312, 0.0295, 0.0236])
                },
            "test":
                {
                    "data": np.array([9755]),
                    "label": np.array([0.0118])
                },
        },
    "hymenoptera_apoidea":
        {
            "train":
                {
                    "data": np.array([216291, 199444, 181639, 198853]),
                    "label": np.array([1.0742, 0.5405, 0.6877, 0.4001])
                },
            "test":
                {
                    "data": np.array([154295]),
                    "label": np.array([0.3124])
                },
        },
    "hymenoptera_formicidae":
        {
            "train":
                {
                    "data": np.array([153965, 27947, 56192, 23901]),
                    "label": np.array([0.0324, 0.1388, 0.0456, 0.1421])
                },
            "test":
                {
                    "data": np.array([56701]),
                    "label": np.array([0.2229])
                },
        },
    "hymenoptera_unknown":
        {
            "train":
                {
                    "data": np.array([112547, 155582, 189349, 303065]),
                    "label": np.array([0.2623, 0.5905, 0.7517, 0.4497])
                },
            "test":
                {
                    "data": np.array([219222]),
                    "label": np.array([0.486])
                },
        },
    "larvae":
        {
            "train":
                {
                    "data": np.array([16813, 57561, 940058, 38251]),
                    "label": np.array([0.0502, 0.298, 6.5697, 0.1333])
                },
            "test":
                {
                    "data": np.array([28857]),
                    "label": np.array([0.0671])
                },
        },
    "lepidoptera":
        {
            "train":
                {
                    "data": np.array([450590, 839512, 209040, 284626]),
                    "label": np.array([1.143, 1.4489, 0.4307, 0.2218])
                },
            "test":
                {
                    "data": np.array([506235]),
                    "label": np.array([1.379])
                },
        },
    "mites":
        {
            "train":
                {
                    "data": np.array([397, 2855, 546, 228]),
                    "label": np.array([0.0001178, 0.000531, 0.0001164, 0.0000257])
                },
            "test":
                {
                    "data": np.array([837]),
                    "label": np.array([0.0001255])
                },
        },
    "neuroptera":
        {
            "train":
                {
                    "data": np.array([64139, 24674]),
                    "label": np.array([0.0615, 0.0189])
                },
            "test":
                {
                    "data": np.array([19085]),
                    "label": np.array([0.0131])
                },
        },
    "non-miteArachnid":
        {
            "train":
                {
                    "data": np.array([41213, 170782, 297217, 108197, 298876]),
                    "label": np.array([0.0837, 0.516, 0.974, 0.3553, 1.0009])
                },
            "test":
                {
                    "data": np.array([168134]),
                    "label": np.array([0.3836])
                },
        },
    "orthoptera":
        {
            "train":
                {
                    "data": np.array([515982, 72608, 78427]),
                    "label": np.array([3.1466, 0.1621, 0.2315])
                },
            "test":
                {
                    "data": np.array([112712]),
                    "label": np.array([0.4973])
                },
        },
    "plecoptera":
        {
            "train":
                {
                    "data": np.array([500370, 66593]),
                    "label": np.array([1.2078, 0.0877])
                },
            "test":
                {
                    "data": np.array([150912]),
                    "label": np.array([0.1876])
                },
        },
    "psocodea":
        {
            "train":
                {
                    "data": np.array([14040, 15578, 8033, 13526]),
                    "label": np.array([0.0144, 0.026, 0.0118, 0.0459])
                },
            "test":
                {
                    "data": np.array([30674]),
                    "label": np.array([0.027])
                },
        },
    "thysanoptera":
        {
            "train":
                {
                    "data": np.array([5191, 4277, 1829, 1204]),
                    "label": np.array([0.002288, 0.0004264, 0.0002374, 0.0001779])
                },
            "test":
                {
                    "data": np.array([2506]),
                    "label": np.array([0.0002995])
                },
        },
    }

def updatePredDict(model_dict, total, error, precentError, order):
    model_dict["order"][order] = total
    model_dict["total"] += total
    model_dict["error"] += error
    model_dict["percentError"] += percentError
    return copy.deepcopy(model_dict)

def getMetrics(order, metric, pred, label):
    error = np.abs(label - pred)
    percentError = (error/label) * 100
    print(f"Avg {metric}", order, label, round(pred, 4), round(error, 4), round(percentError, 4))
    return pred, error, percentError

def ratio(order, train, test):
    totalPixels, totalWeight = 0, 0
    for i in range(len(train["data"])):
        totalPixels += train["data"][i]
        totalWeight += train["label"][i]
    densityPerPixel = totalWeight / totalPixels
    pred = test["data"][0] * densityPerPixel
    return getMetrics(order, "ratio", pred, test["label"][0])

def averageRatio(order, train, test):
    densityPerPixel = 0
    for i in range(len(train["data"])):
        densityPerPixel += train["label"][i] / train["data"][i]
    densityPerPixel /= len(train["data"])
    pred = test["data"][0] * densityPerPixel
    return getMetrics(order, "avgRatio", pred, test["label"][0])

def linearRegression(order, train, test):
    reg = linear_model.LinearRegression()
    reg.fit(train["data"].reshape(-1, 1), train["label"])
    pred = reg.predict(test["data"].reshape(-1, 1))
    # print(f"y={reg.coef_[0]}x + {round(reg.intercept_, 3)}")
    return getMetrics(order, "linReg", pred[0], test["label"][0])

def kNNRegression(order, train, test):
    knn = KNeighborsRegressor(n_neighbors=1)
    knn.fit(train["data"].reshape(-1, 1), train["label"])
    pred = knn.predict(test["data"].reshape(-1, 1))
    return getMetrics(order, "kNNReg", pred[0], test["label"][0])

def svmRegression(order, train, test):
    svmReg = svm.SVR(gamma="auto")
    svmReg.fit(train["data"].reshape(-1, 1), train["label"])
    pred = svmReg.predict(test["data"].reshape(-1, 1))
    return getMetrics(order, "svmReg", pred[0], test["label"][0])

def boxCox(order, train, test):
    data = power_transform(train["data"].reshape(-1, 1), method="box-cox")
    reg = linear_model.LinearRegression()
    reg.fit(data, train["label"])
    print(test["data"])
    pred = reg.predict(power_transform(test["data"].reshape(-1, 1), method="box-cox"))
    return getMetrics(order, "boxCox", pred[0], test["label"][0])

def polyReg(order, train, test, degree):
    model = np.poly1d(np.polyfit(train["data"], train["label"], degree))
    pred = model(test["data"])
    return getMetrics(order, f"polyReg{degree}", pred[0], test["label"][0])

def logReg(order, train, test):
    data = np.log(train["data"])
    reg = linear_model.LinearRegression()
    reg.fit(data.reshape(-1, 1), train["label"])
    pred = reg.predict(np.log(test["data"]).reshape(-1, 1))
    # print(f"y={reg.coef_[0]}x + {round(reg.intercept_, 3)}")
    return getMetrics(order, "logReg", pred[0], test["label"][0])



pred_dict = {

    "ratio": {
        "order": order_dict,
        "total": 0,
        "error": 0,
        "percentError": 0,
    },
    "avgRatio": {
        "order": order_dict,
        "total": 0,
        "error": 0,
        "percentError": 0,
    },
    "linReg": {
        "order": order_dict,
        "total": 0,
        "error": 0,
        "percentError": 0,
    },
    "logReg": {
        "order": order_dict,
        "total": 0,
        "error": 0,
        "percentError": 0,
    },
    "boxCox": {
        "order": order_dict,
        "total": 0,
        "error": 0,
        "percentError": 0,
    },
    "kNNReg": {
        "order": order_dict,
        "total": 0,
        "error": 0,
        "percentError": 0,
    },
    "svmReg": {
        "order": order_dict,
        "total": 0,
        "error": 0,
        "percentError": 0,
    },
    "secDegReg": {
        "order": order_dict,
        "total": 0,
        "error": 0,
        "percentError": 0,
    },
    "thirdDegReg": {
        "order": order_dict,
        "total": 0,
        "error": 0,
        "percentError": 0,
    },
}



for order in data:
    total, error, percentError = ratio(order, data[order]["train"], data[order]["test"])
    pred_dict["ratio"] = updatePredDict(pred_dict["ratio"], total, error, percentError, order)
    total, error, percentError = averageRatio(order, data[order]["train"], data[order]["test"])
    pred_dict["avgRatio"] = updatePredDict(pred_dict["avgRatio"], total, error, percentError, order)
    total, error, percentError = linearRegression(order, data[order]["train"], data[order]["test"])
    pred_dict["linReg"] = updatePredDict(pred_dict["linReg"], total, error, percentError, order)
    total, error, percentError = logReg(order, data[order]["train"], data[order]["test"])
    pred_dict["logReg"] = updatePredDict(pred_dict["logReg"], total, error, percentError, order)
    # total, error, percentError = boxCox(order, data[order]["train"], data[order]["test"])
    # pred_dict["boxCox"] = updatePredDict(pred_dict["boxCox"], total, error, percentError, order)
    total, error, percentError = kNNRegression(order, data[order]["train"], data[order]["test"])
    pred_dict["kNNReg"] = updatePredDict(pred_dict["kNNReg"], total, error, percentError, order)
    total, error, percentError = svmRegression(order, data[order]["train"], data[order]["test"])
    pred_dict["svmReg"] = updatePredDict(pred_dict["svmReg"], total, error, percentError, order)
    # pred_dict["secDegReg"] = updatePredDict(pred_dict["secDegReg"], total, error, percentError)
    # total, error, percentError = polyReg(order, data[order]["train"], data[order]["test"], degree=3)
    # pred_dict["thirdDegReg"] = updatePredDict(pred_dict["thirdDegReg"], total, error, percentError)
    # total, error, percentError = boxCox(order, data[order]["train"], data[order]["test"])


total = 5.09977

for model in pred_dict:
    pred_dict[model]["percentError"] /= len(order_dict)
    print(model,
          total,
          pred_dict[model]["total"],
          np.abs(total - pred_dict[model]["total"]),
          (np.abs(total - pred_dict[model]["total"]) / total) * 100,
          pred_dict[model]["error"],
          pred_dict[model]["percentError"])


model_lst = ["ratio",
    "avgRatio",
    "linReg",
    "logReg",
    "boxCox",
    "kNNReg",
    "svmReg"]

print(pred_dict["ratio"]["order"]['araneae'], )

for key in order_dict:
    # string = f"{key} & {np.format_float_scientific(data[key]['test']['label'][0], precision=2)} & "
    string = f"{key} & {round(data[key]['test']['label'][0], 3)} & "
    min_diff = 1000
    low_model = ""
    for model in model_lst:
        # import ipdb;ipdb.set_trace()
        # string += f"{np.format_float_scientific(pred_dict[model]['order'][key], precision=2)} & "
        # string += f"{round(pred_dict[model]['order'][key], 3)} & "
        string += f"{np.abs(data[key]['test']['label'][0] - pred_dict[model]['order'][key])} & "
        diff = np.abs(data[key]['test']['label'][0] - pred_dict[model]['order'][key])
        # print(diff, min_diff)
        if diff < min_diff:
            min_diff = diff
            low_model = model
        # print(diff, min_diff, low_model)
    print(string)
    print(key, low_model, pred_dict[low_model]['order'][key])

string = f"Total & & "
for model in model_lst:
    string += f"{round(pred_dict[model]['total'], 3)} & "
print(string)










