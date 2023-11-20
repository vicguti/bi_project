import warnings
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

train = pd.read_csv("....../Walmart/train.csv")
test = pd.read_csv("......./Walmart/test.csv")
store = pd.read_csv("....../Walmart/stores.csv")
feature = pd.read_csv("...../Walmart/features.csv")

train.head()


feature.head()


merge_df = pd.merge(train, feature, on=["Store", "Date"], how="inner")


merge_df.head()


merge_df.describe().transpose()


merge_df["DateTimeObj"] = [dt.strptime(x, "%Y-%m-%d") for x in list(merge_df["Date"])]
merge_df["DateTimeObj"].head()


plt.plot(
    merge_df[(merge_df.Store == 1)].DateTimeObj,
    merge_df[(merge_df.Store == 1)].Weekly_Sales,
    "ro",
)
plt.show()


weeklysales = merge_df.groupby(["Store", "Date"])["Weekly_Sales"].apply(
    lambda x: np.sum(x)
)
weeklysales[0:5]


weeklysaledept = merge_df.groupby(["Store", "Dept"])["Weekly_Sales"].apply(
    lambda x: np.sum(x)
)
weeklysaledept[0:5]


weeklyscale = weeklysales.reset_index()
weeklyscale[0:5]


walmartstore = pd.merge(weeklyscale, feature, on=["Store", "Date"], how="inner")
walmartstore.head()


walmartstoredf = walmartstore.iloc[:, list(range(5)) + list(range(10, 13))]


walmartstoredf.head()


walmartstoredf["DateTimeObj"] = [
    dt.strptime(x, "%Y-%m-%d") for x in list(walmartstoredf["Date"])
]
weekNo = walmartstoredf.reset_index()


weekNo = [
    (x - walmartstoredf["DateTimeObj"][0]) for x in list(walmartstoredf["DateTimeObj"])
]


walmartstoredf["Week"] = [np.timedelta64(x, "D").astype(int) / 7 for x in weekNo]


walmartstoredf.head()


plt.plot(walmartstoredf.DateTimeObj, walmartstoredf.Weekly_Sales, "ro")
plt.show()


walmartstoredf["IsHolidayInt"] = [int(x) for x in list(walmartstoredf.IsHoliday)]


walmartstoredf.head()


walmartstoredf.Store.unique()


train_WM, test_WM = train_test_split(walmartstoredf, test_size=0.3, random_state=42)


plt.plot(
    walmartstoredf[(walmartstoredf.Store == 1)].Week,
    walmartstoredf[(walmartstoredf.Store == 1)].Weekly_Sales,
    "ro",
)
plt.show()


XTrain = train_WM[
    ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Week", "IsHolidayInt"]
]
YTrain = train_WM["Weekly_Sales"]


XTest = test_WM[
    ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Week", "IsHolidayInt"]
]
YTest = test_WM["Weekly_Sales"]


wmLinear = linear_model.LinearRegression(normalize=True)
wmLinear.fit(XTrain, YTrain)


wmLinear.coef_


# Performance on the test data sets
YHatTest = wmLinear.predict(XTest)


plt.plot(YTest, YHatTest, "ro")
plt.plot(YTest, YTest, "b-")
plt.show()


walmartstoredf["Store"].unique()


Store_Dummies = pd.get_dummies(walmartstoredf.Store, prefix="Store").iloc[:, 1:]
walmartstoredf = pd.concat([walmartstoredf, Store_Dummies], axis=1)


walmartstoredf.head()


train_WM, test_WM = train_test_split(walmartstoredf, test_size=0.3, random_state=42)
XTrain = train_WM.iloc[
    :, ([3, 4, 5, 6] + [9, 10]) + list(range(11, walmartstoredf.shape[1]))
]
yTrain = train_WM.Weekly_Sales

XTest = test_WM.iloc[
    :, ([3, 4, 5, 6] + [9, 10]) + list(range(11, walmartstoredf.shape[1]))
]
yTest = test_WM.Weekly_Sales


XTrain.head()


wmLinear = linear_model.LinearRegression(normalize=True)
wmLinear.fit(XTrain, YTrain)


# Performance on the test data sets
YHatTest = wmLinear.predict(XTest)
plt.plot(YTest, YHatTest, "ro")
plt.plot(YTest, YTest, "b-")
plt.show()


# calculate the accuray of the model by sum of Square and mean absolute prediction error
MAPE = np.mean(abs((YTest - YHatTest) / YTest))
MSSE = np.mean(np.square(YHatTest - YTest))

print(MAPE, MSSE)


# Dimensionality Reduction


alphas = np.linspace(10, 20, 10)


testError = np.empty(10)

for i, alpha in enumerate(alphas):

    lasso = Lasso(alpha=alpha)
    lasso.fit(XTrain, YTrain)
    testError[i] = mean_squared_error(YTest, lasso.predict(XTest))


plt.plot(alphas, testError, "r-")
plt.show()


wmLinear = linear_model.LinearRegression(normalize=True)
wmLinear


lasso = Lasso(alpha=17)
lasso.fit(XTrain, YTrain)
