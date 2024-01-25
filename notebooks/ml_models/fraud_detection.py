# This notebooks generates a cancer detection model for healthcare

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#Loading data for ML model
df_breast_cancer = pd.read_csv("./data/ml_cases/breast_cancer.csv")

df_breast_cancer_id_out = df_breast_cancer.drop(["id", "Unnamed: 32"], axis=1)

df_breast_cancer_id_out["diagnosis"].replace(["M", "B"], [1, 0], inplace=True)

print(df_breast_cancer_id_out.info())

print(df_breast_cancer_id_out.corr())


# Select the fith main features
from sklearn.feature_selection import SelectKBest, chi2

X = df_breast_cancer_id_out.iloc[:, 1:]
Y = df_breast_cancer_id_out.iloc[:, 0]

best_features = SelectKBest(score_func=chi2, k=5)
fit = best_features.fit(X, Y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
features_scores = pd.concat([df_columns, df_scores], axis=1)
features_scores.columns = ["Features", "Score"]
main_features = features_scores.sort_values(by="Score", ascending=False).iloc[:5, 0]


# Create logistic model
X = df_breast_cancer_id_out[main_features]
Y = df_breast_cancer_id_out[["diagnosis"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.8, random_state=100
)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(X_test)
print(y_pred)

from sklearn import metrics

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Recall: ", metrics.recall_score(y_test, y_pred, zero_division=1))
print("Precision: ", metrics.precision_score(y_test, y_pred, zero_division=1))
print("CL Report: ", metrics.classification_report(y_test, y_pred, zero_division=1))
