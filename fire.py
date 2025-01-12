import cv2
from ultralytics import YOLO
import cvzone
import pandas as pd
import time

# Load model and class names
model = YOLO("../fire/best.pt")  # Adjust path if needed
classNames = ['fire']

# Open video source (adjust path or use webcam as needed)
cap = cv2.VideoCapture("7543653-hd_1920_1080_30fps.mp4")

# Create DataFrame for storing detections
df = pd.DataFrame(columns=['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

# Initialize variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

# Set desired width and height for resizing
desired_width = 740
desired_height = 560

while True:
    new_frame_time = time.time()

    # Capture frame
    success, img = cap.read()

    # If the video ends, reset the video capture to loop again
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the start
        continue  # Skip the rest of the loop and reload the frame

    # Perform object detection
    results = model(img, stream=True)

    for r in results:
        for box in r.boxes:
            # Extract bounding box coordinates, confidence, and class
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            try:
                # Handle potential class index out of range
                if 0 <= cls < len(classNames):
                    # Draw bounding box with green color and thicker lines
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Green color (BGR format), thickness=4

                    # Display label with red text and white background
                    label = f'{classNames[cls]} {conf:.2f}'
                    font_scale = 1.5  # Increase font size
                    thickness = 2  # Increase thickness
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                    # Draw white background rectangle for the text
                    cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1 - 10), (255, 255, 255), -1)  # White background

                    # Put red text on the white background
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                else:
                    print(f"Warning: Class index {cls} out of range. Using default label.")
                    # Draw default label with red text and white background
                    label = "Unknown Class"
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1 - 10), (255, 255, 255), -1)  # White background
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

                # Add detection to DataFrame
                df = pd.concat([df, pd.DataFrame(
                    {'Class': classNames[cls], 'Confidence': conf, 'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2}, index=[0])],
                               ignore_index=True)
            except IndexError:
                print("Error: Index out of range. Skipping detection.")

    # Resize the image for displaying
    img_resized = cv2.resize(img, (desired_width, desired_height))

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img_resized, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the resized frame
    cv2.imshow("Image", img_resized)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Write detections to Excel file
df.to_excel('detections2.xlsx', index=False)

# Release resources
cap.release()
cv2.destroyAllWindows()