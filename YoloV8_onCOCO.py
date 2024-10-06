import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time
from collections import defaultdict
from threading import Thread

# Thread for video capture to improve performance by separating video acquisition from processing
class VideoStream:
    def __init__(self, src=0):
        # Initialize the video capture from the specified source (0 = default camera)
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        # Start the video stream in a separate thread
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Continuously capture frames from the video source until stopped
        while not self.stopped:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()

    def read(self):
        # Return the latest frame captured
        return self.ret, self.frame

    def stop(self):
        # Stop the video stream and release the capture device
        self.stopped = True
        self.cap.release()

# Initialize YOLO model with the detection task
model = YOLO('240_yolov8n_full_integer_quant_edgetpu.tflite', task='detect')

# Start video stream in a separate thread
vs = VideoStream().start()

# Add a short delay to allow the camera to initialize
time.sleep(3)

# Load the COCO class list from a text file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize variables for tracking frame count and time
frame_count = 0
start_time = time.time()

# Variables to track total inference time and the number of frames processed for inference
total_inference_time = 0
inference_frame_count = 0

while True:
    # Read the current frame from the video stream
    ret, frame = vs.read()
    if not ret:
        print("Error: unable to capture frame")
        break

    frame_count += 1
    if frame_count % 6 != 0:
        continue  # Skip every 5 frames to improve performance (infer on every 6th frame)

    # Resize the frame to 240p (320x240) for faster processing
    resized_frame = cv2.resize(frame, (320, 240))

    # Start timer for the inference process
    inference_start = time.time()

    # Run YOLO prediction on the resized frame (240p)
    results = model.predict(resized_frame, imgsz=240)

    # End timer for the inference
    inference_end = time.time()
    total_inference_time += (inference_end - inference_start)
    inference_frame_count += 1

    # Dictionary to count detected objects by class
    label_count = defaultdict(int)

    # Check if there are any detection results
    if len(results) > 0:
        a = results[0].boxes.data
        if a is not None and len(a) > 0:
            px = pd.DataFrame(a).astype("float")

            # Loop through the detection results and draw rectangles around detected objects
            for index, row in px.iterrows():
                # Convert the bounding box coordinates back to the original frame size
                x1 = int(row[0] * (frame.shape[1] / 320))
                y1 = int(row[1] * (frame.shape[0] / 240))
                x2 = int(row[2] * (frame.shape[1] / 320))
                y2 = int(row[3] * (frame.shape[0] / 240))
                d = int(row[5])
                c = class_list[d]

                # Draw rectangles around detected objects and display the class label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

                # Increment the count for the detected class
                label_count[c] += 1

    # Calculate and display FPS (frames per second) for the visual stream
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time
    cvzone.putTextRect(frame, f'FPS (visual): {round(fps, 2)}', (10, 30), 1, 1)

    # Calculate and display FPS for the inference process
    if inference_frame_count > 0:
        inference_fps = inference_frame_count / total_inference_time
        cvzone.putTextRect(frame, f'FPS (inference): {round(inference_fps, 2)}', (10, 60), 1, 1)

    # Display the count of detected objects per class on the frame
    y_offset = 90  # Initial position for the text
    for label, count in label_count.items():
        cvzone.putTextRect(frame, f'{count} {label}', (10, y_offset), 1, 1)
        y_offset += 30  # Move the text down for each new label

    # Show the frame with detections, FPS, and object counts
    cv2.imshow("FRAME", frame)

    # Exit the loop if the ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Stop the video stream and close the display window
vs.stop()
cv2.destroyAllWindows()
