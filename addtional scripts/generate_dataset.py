import cv2
import os

SAVE_DIR = "marked" 
BOX_SIZE = 300

# Create folder if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

img_count = 78

print("===================================")
print("Left click to save image")
print("Press 'q' to quit")
print("===================================")

# Global frame storage
current_crop = None

# Mouse callback
def mouse_callback(event, x, y, flags, param):
    global img_count, current_crop

    # Left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:

        if current_crop is not None:

            filename = os.path.join(
                SAVE_DIR,
                f"img_{img_count}.jpg"
            )

            cv2.imwrite(filename, current_crop)

            print(f"Saved: {filename}")

            img_count += 1

# Create window
cv2.namedWindow("Bottle Dataset Capture")

# Attach mouse callback
cv2.setMouseCallback("Bottle Dataset Capture", mouse_callback)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Center box coordinates
    x1 = w // 2 - BOX_SIZE // 2
    y1 = h // 2 - BOX_SIZE // 2
    x2 = x1 + BOX_SIZE
    y2 = y1 + BOX_SIZE

    # Draw guide box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Instructions
    cv2.putText(
        frame,
        "Place bottle inside box",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Left click to save",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Crop box area
    current_crop = frame[y1:y2, x1:x2]

    # Show webcam
    cv2.imshow("Bottle Dataset Capture", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()