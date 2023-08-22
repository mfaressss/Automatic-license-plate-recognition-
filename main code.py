import torch
import cv2
from easyocr import Reader

# Load the model
model_path = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, verbose=True)

# Set the model to evaluation mode
model.eval()

# Initialize EasyOCR reader
reader = Reader(['en'], gpu=False, verbose=False)

# Open the camera (change the index if you have multiple cameras)
camera = cv2.VideoCapture(0)  # 0 for default camera

while True:
    # Capture a frame from the camera
    ret, frame = camera.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Process detection results (bounding boxes, etc.)
    for det in results.pred:
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f'Class {int(cls)} - Confidence: {conf:.2f}'

            # Extract license plate region
            plate = frame[y1:y2, x1:x2]

            # Perform OCR on license plate
            ocr_results = reader.readtext(plate)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Add label

            # Display OCR results
            if ocr_results:
                plate_text = ocr_results[0][1]
                cv2.putText(frame, f'Plate: {plate_text}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes and OCR text
    cv2.imshow('YOLOv5 Object Detection with OCR', frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
