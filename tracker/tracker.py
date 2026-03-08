import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.models as models
import torchvision.transforms as transforms

yolo = YOLO("yolov8n.pt")
resnet = models.resnet18(pretrained=True)
feature_net = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_net.eval()

SIMILARITY_THRESHOLD = 0.75
TRACK_UPDATE_CONFIDENCE = 0.85
ADAPTATION_RATE = 0.3

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_features(img):
    if img.size == 0:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = feature_net(img).squeeze().numpy()
    return vec / (np.linalg.norm(vec) + 1e-6)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera error")
    exit()

print("Press S to select object")

target = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    if target is None:
        cv2.putText(display, "Press S to select object", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        results = yolo(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []

        best_box, best_score, best_vec = None, -1, None

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            vec = get_features(crop)
            if vec is None:
                continue
            score = np.dot(target, vec)
            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)
                best_vec = vec

        if best_box and best_score > SIMILARITY_THRESHOLD:
            cv2.rectangle(display, best_box[:2], best_box[2:], (0, 255, 0), 3)
            cv2.putText(display, f"Track {best_score:.2f}",
                        (best_box[0], best_box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if best_score > TRACK_UPDATE_CONFIDENCE and best_vec is not None:
                target = ADAPTATION_RATE * best_vec + (1 - ADAPTATION_RATE) * target
                target /= (np.linalg.norm(target) + 1e-6)
        else:
            cv2.putText(display, "Object lost", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Object Tracker", display)
    key = cv2.waitKey(1)

    if key == ord('s'):
        box = cv2.selectROI("Select Object", frame, False)
        cv2.destroyWindow("Select Object")
        x, y, w, h = box
        if w > 0 and h > 0:
            crop = frame[int(y):int(y+h), int(x):int(x+w)]
            vec = get_features(crop)
            if vec is not None:
                target = vec
                print("Selected")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
