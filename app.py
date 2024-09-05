from flask import Flask, render_template, Response
import cv2
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

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

def generate_video():
    global car_count, previous_cars
    car_count = 0
    previous_cars = []

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        height, width, _ = frame.shape

        # Prepare the image for the YOLO model
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Forward pass
        detections = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for detection in detections:
            for object_detection in detection:
                scores = object_detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and class_id == CAR_CLASS_ID:
                    center_x = int(object_detection[0] * width)
                    center_y = int(object_detection[1] * height)
                    w = int(object_detection[2] * width)
                    h = int(object_detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        current_cars = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                if class_ids[i] == CAR_CLASS_ID:
                    current_cars.append([x, y, w, h])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for current_car in current_cars:
            new_car = True
            for previous_car in previous_cars:
                iou = calculate_iou(current_car, previous_car)
                if iou > overlap_threshold:
                    new_car = False
                    break

            if new_car:
                car_count += 1
                previous_cars.append(current_car)

        cv2.putText(frame, f"Total Cars Detected: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame to Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
