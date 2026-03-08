import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights

# Load YOLO model
yolo = YOLO("yolov8n.pt")

# Load ResNet18 as feature extractor
resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
feature_net = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_net.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# Convert image to feature vector
def extract_features(img):
    with torch.no_grad():
        img = transform(img).unsqueeze(0)
        vec = feature_net(img).squeeze().numpy()
        vec = vec / (np.linalg.norm(vec) + 1e-6)
    return vec

# Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected")
    exit()

print("Press S to select object")

target_vector = None

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # If object not selected yet
    if target_vector is None:

        cv2.putText(frame,
                    "Press S to select object",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    2)

    else:

        # Detect objects with YOLO
        result = yolo(frame)[0]
        boxes = result.boxes.xyxy.cpu().numpy()

        best_box = None
        best_score = -1

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            vec = extract_features(crop)

            # Cosine similarity
            score = np.dot(target_vector, vec)

            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)

        # Draw best match
        if best_box and best_score > 0.682:

            cv2.rectangle(frame,
                          best_box[:2],
                          best_box[2:],
                          (0,255,0),
                          3)

            cv2.putText(frame,
                        "Tracking",
                        (best_box[0], best_box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

        else:
           

    cv2.imshow("Object Tracker", frame)

    key = cv2.waitKey(1)

    # Select object
    if key == ord('s'):

        box = cv2.selectROI("Select Object", frame, False)
        cv2.destroyWindow("Select Object")

        x,y,w,h = box
        crop = frame[int(y):int(y+h), int(x):int(x+w)]

        if crop.size != 0:
            target_vector = extract_features(crop)

    # Exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


