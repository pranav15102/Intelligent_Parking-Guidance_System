import cv2
import torch
import numpy as np
import pickle
from shapely.geometry import Point, Polygon
import csv
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load video
input_path = "parking2.mp4"
cap = cv2.VideoCapture(input_path)

regions = "regions2.p"
with open(regions, 'rb') as f:
    parking_boxes = pickle.load(f)

def calculate_iou(box, polygon_coordinates):
    x_min, y_min, x_max, y_max = box
    box_polygon = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
    parking_polygon = Polygon(polygon_coordinates)
    intersection = box_polygon.intersection(parking_polygon).area
    union = box_polygon.union(parking_polygon).area
    iou = intersection / union if union != 0 else 0
    return iou

class ParkingSpace:
    _counter = 0

    def __init__(self, region):
        ParkingSpace._counter += 1
        self.id = ParkingSpace._counter
        self.name = region["name"]
        self.vertices = np.array(region["points"], np.int32).tolist()
        polygon = Polygon(self.vertices)
        polygon_center = polygon.centroid.xy
        self.center_x, self.center_y = polygon_center[0][0], polygon_center[1][0]
        self.occupancy_stat = False
        self.iou = 0.0

psObjs = [ParkingSpace(region) for region in parking_boxes]

# Updating GPS Coordinates
csv_file = 'Final GPS.csv'
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        for ps in psObjs:
            if row[0].replace(" ", "") == ps.name:
                ps.lat = float(row[1])
                ps.long = float(row[2])
                break

last_frame = None
occupancy_results = {}
iou_results = {}

iou_threshold = 0.45  # Set IoU threshold for occupancy

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to grab frame")
        break

    results = model(frame)
    car_boxes = results.xyxy[0].cpu().numpy()
    vehicle_classes = [2, 3, 5, 7]
    car_boxes = car_boxes[np.isin(car_boxes[:, 5], vehicle_classes)]
    car_boxes = car_boxes[car_boxes[:, 4] >= 0.25]

    for ps in psObjs:
        ps.occupancy_stat = False
        ps.iou = 0.0

        for box in car_boxes:
            x_min, y_min, x_max, y_max, confidence, class_label = box
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 4)

            iou = calculate_iou((x_min, y_min, x_max, y_max), ps.vertices)
            ps.iou = max(ps.iou, iou)

        # Determine occupancy based on IoU threshold
        if ps.iou > iou_threshold:
            ps.occupancy_stat = True

        occupancy_results[ps.name] = ps.occupancy_stat
        iou_results[ps.name] = ps.iou

    for ps in psObjs:
        yc = ps.center_y + 15
        color = (0, 255, 0) if not ps.occupancy_stat else (0, 0, 255)

        cv2.polylines(frame, [np.array(ps.vertices, np.int32)], True, color, 4)


        # cv2.putText(frame, f'IoU: {ps.iou:.2f}', (int(ps.center_x), int(ps.center_y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, ps.name, (int(ps.center_x), int(yc)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('RGB', frame)
    last_frame = frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if last_frame is not None:
    save_path = os.path.join(os.path.dirname(__file__), 'last_frame2.png')
    cv2.imwrite(save_path, last_frame)
    print(f"Last frame saved as {save_path}")

cap.release()
cv2.destroyAllWindows()

results = {
    'occupancy': occupancy_results,
    'iou': iou_results
}
with open('detection2.pkl', 'wb') as f:
    pickle.dump(results, f)



