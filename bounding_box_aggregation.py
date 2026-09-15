import cv2
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from collections import Counter

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

from PIL import Image
from concurrent.futures import ThreadPoolExecutor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# load the individual yolo weather models
# -----------------------------
clear_model = YOLO("models/clear_model.pt")
fog_model = YOLO("models/fog_model.pt")
rain_streaks_model = YOLO("models/rain_streaks.pt")
shadow_sunglare_model = YOLO("models/shadow_sunglare.pt")

model_inputs = [
    ("clear", clear_model),
    ("fog", fog_model),
    ("rain", rain_streaks_model),
    ("shadow", shadow_sunglare_model)
]

class_names = clear_model.names


# -----------------------------
# load the weather classification model 
# -----------------------------
weather_model = models.resnet18(weights=None)
weather_model.fc = nn.Linear(weather_model.fc.in_features, 4)

weather_model = torch.load(
    "models/weather_classification_model.2780701652526447",
    map_location=device,
    weights_only=False
)

weather_model = weather_model.to(device)
weather_model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# load images
# -----------------------------
image_path = "image/rainy3.jpg"
image = cv2.imread(image_path)
height, width, _ = image.shape


# -----------------------------
# run the weather prediction model
# -----------------------------
def predict_weather(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = weather_model(input_tensor)
        probs = torch.softmax(output, dim=1)

    cls = torch.argmax(probs, dim=1).item()
    return ["clear", "fog", "rain", "shadow"][cls]


weather_label = predict_weather(image)
print("Weather prediction:", weather_label)


# -----------------------------
# add weights depending on which weather the model has classified
# -----------------------------
weights = {m: 0.8 for m in ["clear", "fog", "rain", "shadow"]}
weights[weather_label] = 1.6

print("Model weights:", weights)


# -----------------------------
# draw bounding boxes around the detections
# -----------------------------
def run_model(name, model, image, weight, width, height, class_names):

    results = model(image)[0]

    boxes = []
    scores = []
    labels = []
    local_nms_boxes = []
    local_nms_scores = []
    local_nms_labels = []

    img_copy = image.copy()

    

    for box in results.boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        conf = float(box.conf[0]) * weight
        conf = min(conf, 1.0)

        cls = int(box.cls[0])
        label_name = class_names[cls]

        cv2.rectangle(img_copy,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0), 2)

        cv2.putText(img_copy,
                    f"{label_name}:{conf:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2)

        # WBF format
        boxes.append([x1/width, y1/height, x2/width, y2/height])
        scores.append(conf)
        labels.append(cls)

        # NMS format
        local_nms_boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
        local_nms_scores.append(conf)
        local_nms_labels.append(cls)

    return name, boxes, scores, labels, img_copy, local_nms_boxes, local_nms_scores, local_nms_labels


# -----------------------------
# run the yolo models in parralel with each other to decrease time taken from input to output
# -----------------------------
results = []

with ThreadPoolExecutor() as executor:

    futures = []

    for name, model in model_inputs:
        futures.append(
            executor.submit(
                run_model,
                name,
                model,
                image,
                weights[name],
                width,
                height,
                class_names
            )
        )

    for f in futures:
        results.append(f.result())


# -----------------------------
# collecting the results
# -----------------------------
all_boxes, all_scores, all_labels = [], [], []
nms_boxes, nms_scores, nms_labels = [], [], []

for res in results:

    name, boxes, scores, labels, img_copy, nb, ns, nl = res

    all_boxes.append(boxes)
    all_scores.append(scores)
    all_labels.append(labels)

    nms_boxes.extend(nb)
    nms_scores.extend(ns)
    nms_labels.extend(nl)

    cv2.imwrite(f"{name}_model_detections.jpg", img_copy)


# -----------------------------
# Non Maximum Suppression
# -----------------------------
indices = cv2.dnn.NMSBoxes(
    nms_boxes,
    nms_scores,
    score_threshold=0.3,
    nms_threshold=0.5
)

nms_image = image.copy()
nms_class_labels = []

if len(indices) > 0:
    for i in indices.flatten():

        x, y, w, h = nms_boxes[i]
        label = nms_labels[i]
        score = nms_scores[i]

        label_name = class_names.get(label, str(label))
        nms_class_labels.append(label_name)

        cv2.rectangle(nms_image,
                      (x, y),
                      (x + w, y + h),
                      (0, 0, 255), 3)

        cv2.putText(nms_image,
                    f"{label_name}:{score:.2f}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)

print("NMS detection counts:", Counter(nms_class_labels))


# -----------------------------
# WBF method
# -----------------------------
if len(all_boxes) > 0:

    boxes, scores, labels = weighted_boxes_fusion(
        all_boxes,
        all_scores,
        all_labels,
        iou_thr=0.55,
        skip_box_thr=0.3
    )

    boxes = np.array(boxes)

    if len(boxes) > 0:
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height

    wbf_image = image.copy()
    wbf_class_labels = []

    for box, score, label in zip(boxes, scores, labels):

        x1, y1, x2, y2 = box.astype(int)
        label_name = class_names.get(label, str(label))

        wbf_class_labels.append(label_name)

        cv2.rectangle(wbf_image,
                      (x1, y1),
                      (x2, y2),
                      (255, 0, 0), 3)

        cv2.putText(wbf_image,
                    f"{label_name}:{score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2)

    print("WBF detection counts:", Counter(wbf_class_labels))

    cv2.imwrite("wbf_result.jpg", wbf_image)


# -----------------------------
# show the results
# -----------------------------
cv2.imwrite("nms_result.jpg", nms_image)
cv2.imwrite("wbf_result.jpg", wbf_image)

print("Images saved as nms_result.jpg and wbf_result.jpg")