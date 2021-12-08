import pandas as pd
import cv2
from sklearn.metrics import accuracy_score
import numpy as np
from utils import hist_equilise, filter_edge, local_face_points, hog_transform
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def csv_img(df, no):
    vals = df.loc[no]["pixels"].split(" ")
    vals = np.array(vals, dtype="uint8")
    vals = vals.reshape((48, 48))
    return vals


def preprocess_data(df):
    X = list()
    Y = list()
    for index, row in tqdm(df.iterrows(), total=len(df)):
        val = hog_transform(csv_img(df, index))
        X.append(val)
        Y.append(df.loc[index]["emotion"])
    return np.array(X), np.array(Y)


def train(X_train, Y_train):
    print(X_train)
    model = Pipeline(steps=[("svc", SVC(verbose=True, kernel="rbf"))])
    model.fit(X_train, Y_train)
    return model


def test(model, X_test, Y_test):
    preds = model.predict(X_test)
    confusion_mat = confusion_matrix(Y_test, preds)
    confusion_mat = pd.DataFrame(confusion_mat)
    total_accuracy = accuracy_score(Y_test, preds)
    print(confusion_mat)
    cmn = confusion_mat.astype("float") / confusion_mat.sum(axis=1)[:, np.newaxis]
    confusion_mat_percent = pd.DataFrame(cmn)
    print(confusion_mat_percent)

    print(f"Total accuracy score is: {total_accuracy}")


if __name__ == "__main__":
    df = pd.read_csv("preprocessed_dataset.csv")
    train_len = len(df[df["Usage"] == "Training"])
    test_len = len(df[df["Usage"] == "Testing"])
    print(f"train length {train_len}")
    print(f"test length {test_len}")

    X_train, Y_train = preprocess_data(df[df["Usage"] == "Training"])
    model = train(X_train, Y_train)
    labels = {}

    model = {"model": model, "labels": labels}
    joblib.dump(model, "model_2.joblib")
    model = joblib.load("model_2.joblib")
    model, labels = model["model"], model["labels"]
    print("model loaded")

    X_test, Y_test = preprocess_data(df[df["Usage"] == "Testing"].reset_index())

    test(model, X_test, Y_test)
