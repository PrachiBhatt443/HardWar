import numpy as np
import cv2
import time
import os  # Use os to play sound on Raspberry Pi

# Paths to configuration and weight files
labelpath = r'F:/Crowd/coco.names'
file = open(labelpath)
label = file.read().strip().split("\n")

weightspath = r'F:/Crowd/yolov3.weights'
configpath = r'F:/Crowd/yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(configpath, weightspath)

videopath = r'f1.mp4'

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Open video capture
video = cv2.VideoCapture(videopath)

# Grid configuration
grid_size = (4, 4)  # 4x4 grid
grid_density = np.zeros(grid_size)  # Density count for each grid cell

# Crowd detection threshold
crowd_threshold = 10  # Define a threshold for "crowd" detection

# Path to beep sound (Ensure 'beep.wav' is in the correct directory)
beep_sound_path = "./beep.wav"

while True:
    ret, frame = video.read()
    if not ret:
        print('Error running the file :(')
        break

    # Reduce the resolution to speed up processing
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time()

    boxes = []
    confidences = []
    classIDs = []

    h, w = frame.shape[:2]

    # Loop through the YOLO outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5 and label[classID] == 'person':
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Non-maxima suppression to avoid overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    crowd_count = 0
    grid_density.fill(0)  # Reset grid density

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Determine the grid cell the person belongs to
            grid_x = int(x / (w / grid_size[1]))
            grid_y = int(y / (h / grid_size[0]))

            grid_x = min(max(0, grid_x), grid_size[1] - 1)
            grid_y = min(max(0, grid_y), grid_size[0] - 1)

            # Increment density count for the grid cell
            grid_density[grid_y][grid_x] += 1

            # Draw bounding boxes around people
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate total crowd count and density
    crowd_count = np.sum(grid_density)

    # Visualize the grid density
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            density = grid_density[row][col]
            color = (0, int(min(density * 50, 255)), 255 - int(min(density * 50, 255)))  # Color intensity based on density
            grid_x = col * (w // grid_size[1])
            grid_y = row * (h // grid_size[0])
            cv2.rectangle(frame, (grid_x, grid_y), (grid_x + (w // grid_size[1]), grid_y + (h // grid_size[0])), color, 2)

    # Display the crowd count on the frame
    cv2.putText(frame, "Crowd Count: {}".format(int(crowd_count)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # If crowd count exceeds threshold, display a warning message and sound a beep
    if crowd_count >= crowd_threshold:
        cv2.putText(frame, "Crowd Detected!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        # Play beep sound using `aplay`
        os.system(f'aplay {beep_sound_path}')

    # Display the frame
    cv2.imshow('Crowd Detection', frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()
