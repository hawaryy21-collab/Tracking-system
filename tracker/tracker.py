import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.models as models
import torchvision.transforms as transforms

# ----- Models -----
yolo = YOLO("yolov8n.pt")
resnet = models.resnet18(pretrained=True)
feature_net = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_net.eval()

# ----- Parameters (tunable) -----
SIMILARITY_THRESHOLD = 0.75      # minimum score to draw box and show "Track"
TRACK_UPDATE_CONFIDENCE = 0.85   # higher threshold to update the reference vector
EMA_ALPHA = 0.3                  # how fast the reference adapts (0.1–0.3)

# ----- Preprocessing -----
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(img):
    """Extract normalized feature vector from an image crop."""
    if img.size == 0:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0)
    with torch.no_grad():
        vec = feature_net(img_tensor).squeeze().numpy()
    return vec / (np.linalg.norm(vec) + 1e-6)

# ----- Main loop -----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not detected")
    exit()

print("Press S to select object")

target_vector = None   # reference feature vector

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    if target_vector is None:
        cv2.putText(display, "Press S to select object", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        # Run YOLO detection
        results = yolo(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []

        best_box, best_score, best_vec = None, -1, None

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            vec = extract_features(crop)
            if vec is None:
                continue
            score = np.dot(target_vector, vec)   # cosine similarity
            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)
                best_vec = vec

        if best_box is not None and best_score > SIMILARITY_THRESHOLD:
            # Draw green bounding box
            cv2.rectangle(display, best_box[:2], best_box[2:], (0, 255, 0), 3)
            cv2.putText(display, f"Track {best_score:.2f}",
                        (best_box[0], best_box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Update reference vector only if confidence is high (adapts to lighting)
            if best_score > TRACK_UPDATE_CONFIDENCE and best_vec is not None:
                target_vector = EMA_ALPHA * best_vec + (1 - EMA_ALPHA) * target_vector
                target_vector /= (np.linalg.norm(target_vector) + 1e-6)
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
            vec = extract_features(crop)
            if vec is not None:
                target_vector = vec   # initialise with the selected object
                print("Object selected")

    if key == 27:   # ESC
        break

cap.release()
cv2.destroyAllWindows()
