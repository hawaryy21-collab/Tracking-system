import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights

# Load models
yolo = YOLO("yolov8n.pt")
cnn = torch.nn.Sequential(*list(models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1]).eval()

# Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_vector(img):
    with torch.no_grad():
        vec = cnn(transform(img).unsqueeze(0)).squeeze().numpy()
        return vec / (np.linalg.norm(vec) + 1e-6)

# Start capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

ret, frame = cap.read()
if not ret or frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
    print("Error: Failed to capture initial frame. Check camera connection.")
    cap.release()
    exit()

# Select object
bbox = cv2.selectROI("Select", frame, False)
cv2.destroyWindow("Select")

selected_crop = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
if selected_crop.size == 0:
    print("Error: Selected crop is empty.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

selected_vec = get_vector(selected_crop)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = yolo(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    best_box, best_sim = None, -1

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue

        vec = get_vector(crop)
        sim = np.dot(selected_vec, vec)

        if sim > best_sim:
            best_sim = sim
            best_box = (x1, y1, x2, y2)

    if best_box and best_sim > 0.5:
        cv2.rectangle(frame, best_box[:2], best_box[2:], (0, 255, 0), 3)
        cv2.putText(frame, "Tracking", (best_box[0], best_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Object Disappeared", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Tracker", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()