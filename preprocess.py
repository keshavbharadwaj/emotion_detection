import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser


def rename_emotion_labels(x):
    if x == 2:
        return 1
    elif x == 3:
        return 2
    elif x == 4:
        return 3
    elif x == 6:
        return 4
    else:
        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.data_path)
    df = df.drop(["Usage"], axis=1)
    df = df[~df["emotion"].isin([1, 5])].reset_index(drop=True)
    print(df)
    df["emotion"] = df["emotion"].apply(lambda x: rename_emotion_labels(x))
    print(df["emotion"].value_counts())

    x = [[i] for i in range(len(df))]
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        df["emotion"].tolist(),
        test_size=0.33,
        random_state=42,
        stratify=df["emotion"].tolist(),
    )
    X_train = [i[0] for i in X_train]
    X_test = [i[0] for i in X_test]
    train = df.loc[X_train].reset_index(drop=True)
    train["Usage"] = "Training"
    test = df.loc[X_test].reset_index(drop=True)
    test["Usage"] = "Testing"
    df = pd.concat([train, test], ignore_index=True)
    print(df[["Usage", "emotion"]].value_counts())
    df.to_csv("preprocessed_dataset.csv", index=False)
