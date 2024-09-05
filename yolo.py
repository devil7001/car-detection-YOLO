import cv2
import numpy as np

# Load YOLO network
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO names (list of object names YOLO can detect)
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define the class ID for "car" in COCO dataset (ID: 2)
CAR_CLASS_ID = 2

# Initialize the webcam
cap = cv2.VideoCapture(0)

car_count = 0
previous_cars = []  # List to store previously detected car bounding boxes
overlap_threshold = 0.5  # Define overlap threshold

# Function to calculate IoU (Intersection over Union) for bounding boxes
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the intersection rectangle
    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)
    x_inter2 = min(x1 + w1, x2 + w2)
    y_inter2 = min(y1 + h1, y2 + h2)

    # Area of intersection
    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)

    # Areas of the bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Area of union
    union_area = box1_area + box2_area - inter_area

    # IoU (Intersection over Union)
    iou = inter_area / union_area

    return iou

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get the frame's height, width
    height, width, channels = frame.shape

    # Prepare the image for the YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Forward pass
    detections = net.forward(output_layers)

    # Initialize list for bounding boxes, confidences, and class_ids
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each detection
    for detection in detections:
        for object_detection in detection:
            scores = object_detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # If the object detected is a car (class ID: 2) and confidence is high enough
            if confidence > 0.5 and class_id == CAR_CLASS_ID:
                # Object detected
                center_x = int(object_detection[0] * width)
                center_y = int(object_detection[1] * height)
                w = int(object_detection[2] * width)
                h = int(object_detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append to the lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression to eliminate redundant overlapping boxes with lower confidences
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Draw bounding boxes and count cars
    current_cars = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(class_names[class_ids[i]])
            confidence = confidences[i]

            if class_ids[i] == CAR_CLASS_ID:
                # Add current car's bounding box to list
                current_cars.append([x, y, w, h])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Car: {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Check for new cars that were not counted previously
    for current_car in current_cars:
        new_car = True
        for previous_car in previous_cars:
            # Calculate IoU between the current car and previously detected cars
            iou = calculate_iou(current_car, previous_car)
            if iou > overlap_threshold:  # If overlap is greater than threshold, it is the same car
                new_car = False
                break

        # If it's a new car, increase the count
        if new_car:
            car_count += 1
            previous_cars.append(current_car)

    # Limit the size of previous_cars list to avoid memory overflow (if needed)
    if len(previous_cars) > 100:  # Tune the size based on your application
        previous_cars = previous_cars[-100:]

    # Display the car count on the frame
    cv2.putText(frame, f"Total Cars Detected: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("YOLO Car Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
