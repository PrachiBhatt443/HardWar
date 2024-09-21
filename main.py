import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO names (object labels)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get layer names
layer_names = net.getLayerNames()

# Fix for IndexError: invalid index to scalar variable
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:
    # For older versions of OpenCV
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Function to estimate crowd density based on person count and frame area
def get_crowd_density(person_count, frame_area):
    density = person_count / frame_area
    if density < 0.00005:  # Low crowd density
        return "Low"
    elif density < 0.0001:  # Medium crowd density
        return "Medium"
    else:  # High crowd density
        return "High"

# Function to detect crowd in video or camera feed
def detect_crowd(video_source):
    cap = cv2.VideoCapture(video_source)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break  # Break if the frame is not captured properly

        height, width, channels = frame.shape
        frame_area = height * width

        # Preprocess the frame for YOLO input
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0),
            swapRB=True, crop=False
        )

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Analyze detection results
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                # Get scores, class ID, and confidence of the prediction
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter detections by confidence threshold and target class ("person")
                if confidence > 0.5 and classes[class_id] == "person":
                    # Object detected; get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Append to lists
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression to eliminate redundant overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Draw bounding boxes and labels
        font = cv2.FONT_HERSHEY_PLAIN
        count = 0  # Counter for detected persons

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, f"{label} {confidence:.2f}",
                    (x, y - 10), font, 1, color, 2
                )
                count += 1

        # Estimate crowd density
        density_label = get_crowd_density(count, frame_area)

        # Display the number of people detected and crowd density
        cv2.putText(
            frame, f"People Count: {count}",
            (10, 50), font, 2, (255, 0, 0), 3
        )
        cv2.putText(
            frame, f"Crowd Density: {density_label}",
            (10, 100), font, 2, (0, 0, 255), 3
        )

        # Display the frame with detections and density label
        cv2.imshow("Crowd Detection and Density", frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Main logic to choose between camera or video file
def main():
    print("Choose an option:")
    print("1. Detect crowd live through camera")
    print("2. Detect crowd through video file")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '1':
        print("Starting live camera detection...")
        detect_crowd(0)  # 0 is for the default camera
    elif choice == '2':
        video_path = input("Enter the path to the video file: ").strip()
        print(f"Starting video detection from {video_path}...")
        detect_crowd(video_path)
    else:
        print("Invalid choice. Please restart and choose 1 or 2.")

if __name__ == "__main__":
    main()