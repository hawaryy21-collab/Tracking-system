import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights

# -----------------------------
# Models
# -----------------------------

yolo = YOLO("yolov8n.pt")

resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
feature_net = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_net.eval()

# -----------------------------
# Parameters (Tunable)
# -----------------------------

SIMILARITY_THRESHOLD = 0.75
EMA_ALPHA = 0.25        # Vector update rate (0.1–0.3 good)
TRACK_UPDATE_CONFIDENCE = 0.80

# -----------------------------
# Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Feature extractor
# -----------------------------

def extract_features(img):
    with torch.no_grad():
        img = transform(img).unsqueeze(0)
        vec = feature_net(img).squeeze().numpy()

        # Normalize vector
        vec = vec / (np.linalg.norm(vec) + 1e-6)

    return vec

# -----------------------------
# EMA vector update
# -----------------------------

def update_target_vector(old_vec, new_vec, alpha=EMA_ALPHA):
    return alpha * new_vec + (1 - alpha) * old_vec

# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected")
    exit()

print("Press S to select object")

target_vector = None
tracking_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # -------------------------
    # If object not selected
    # -------------------------

    if target_vector is None:
        cv2.putText(
            display_frame,
            "Press S to select object",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    else:
        # -------------------------
        # YOLO Detection
        # -------------------------

        result = yolo(frame)[0]

        boxes = result.boxes.xyxy.cpu().numpy()

        best_box = None
        best_score = -1
        best_vec = None

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            vec = extract_features(crop)

            score = np.dot(target_vector, vec)

            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)
                best_vec = vec

        # -------------------------
        # Tracking Decision
        # -------------------------

        if best_box is not None and best_score > SIMILARITY_THRESHOLD:

            tracking_active = True

            # Draw tracking box
            cv2.rectangle(
                display_frame,
                best_box[:2],
                best_box[2:],
                (0, 255, 0),
                3
            )

            cv2.putText(
                display_frame,
                f"Track {best_score:.2f}",
                (best_box[0], best_box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # -------------------------
            # Update stored vector (EMA memory learning)
            # -------------------------

            if best_vec is not None:
                target_vector = update_target_vector(
                    target_vector,
                    best_vec,
                    EMA_ALPHA
                )

                # Re-normalize
                target_vector = target_vector / (np.linalg.norm(target_vector) + 1e-6)

        else:
            tracking_active = False

            cv2.putText(
                display_frame,
                "Object lost",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    # -------------------------
    # Show frame
    # -------------------------

    cv2.imshow("Object Tracker", display_frame)

    key = cv2.waitKey(1)

    # -------------------------
    # Object selection (press S)
    # -------------------------

    if key == ord('s'):
        box = cv2.selectROI("Select Object", frame, False)
        cv2.destroyWindow("Select Object")

        x, y, w, h = box
        crop = frame[int(y):int(y+h), int(x):int(x+w)]

        if crop.size != 0:
            target_vector = extract_features(crop)
            tracking_active = True

            print("Object selected")

    # Exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
