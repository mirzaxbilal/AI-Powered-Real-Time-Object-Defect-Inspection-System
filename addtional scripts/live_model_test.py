import cv2
from ultralytics import YOLO

# mdoel link goes here
model = YOLO("./runs/classify/runs/classify/bottle_classifier-2/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Press q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip webcam for mirror effect
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Draw center rectangle guide
    box_size = 250
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]

    # Match training size
    results = model.predict(roi, imgsz=640, verbose=False)

    probs = results[0].probs
    class_id = probs.top1
    confidence = probs.top1conf.item()
    class_name = results[0].names[class_id]

    label = f"{class_name}: {confidence:.2f}"

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Bottle Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
