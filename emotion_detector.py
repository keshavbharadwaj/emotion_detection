import cv2
from argparse import ArgumentParser
import joblib
import numpy as np
from time import time
import torch
from facenet_pytorch import MTCNN
from tensorflow import keras

emotion_label_to_text = {
    0: "anger",
    1: "fear",
    2: "happiness",
    3: "sadness",
    4: "neutral",
}


device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)
model = keras.models.load_model("model.h5")
print("Model loaded successfully")


def full_flow(img):
    if len(img) == 0:
        return img
    inf_mtcnn_start = time()
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    if boxes is None:
        return None
    inf_mtcnn_end = time()
    # print(f"inference time mtcnn {inf_mtcnn_end - inf_mtcnn_start}")

    for box in boxes:
        neg = False
        for i in box:
            if i < 0:
                neg = True
                break
        if box is None or neg:
            continue
        print(box)
        x_left = int(min(box[0], box[2]))
        x_right = int(max(box[0], box[2]))
        y_left = int(min(box[1], box[3]))
        y_right = int(max(box[1], box[3]))
        center = ((x_left + x_right) // 2, (y_left + y_right) // 2)
        img = cv2.rectangle(
            img,
            (x_left, y_left),
            (x_right, y_right),
            (255, 0, 0),
            2,
        )
        cropped_image = img[y_left:y_right, x_left:x_right]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cropped_image = cv2.resize(
            cropped_image, (48, 48), interpolation=cv2.INTER_AREA
        )
        cropped_image = cropped_image / 255.0
        cropped_image = np.expand_dims(cropped_image, axis=0)
        prediction = model.predict(cropped_image)
        class_index = int(np.where(prediction[0] == max(prediction[0]))[0])
        pred_class = emotion_label_to_text[class_index]
        cv2.putText(
            img,
            pred_class,
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )

    return img


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_path", default=None)
    args = parser.parse_args()
    if args.img_path is not None:
        img = cv2.imread(args.img_path)
        print("image_loaded")
        output = full_flow(img)

        cv2.imshow("output", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        while True:
            ret, frame = cap.read()
            output = full_flow(frame)
            if output is None:
                cv2.imshow("output", frame)
            else:
                cv2.imshow("output", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
