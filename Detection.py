import cv2
import torch
import numpy as np
import pickle
from shapely.geometry import Point, Polygon
import json
import csv
import pymongo
from pymongo import MongoClient

connection_string = "mongodb+srv://svissava:Manojmongo7@cluster0.z8fr6ar.mongodb.net/IPS"
client = MongoClient(connection_string)

# Replace "your_database" and "your_collection" with the desired database and collection names
database = client.IPS
collection = database["parking-space"]

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        # print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load image or video
input_path = "parkingvideo.mp4"
cap = cv2.VideoCapture(input_path)
# cap = cv2.VideoCapture(42)

regions = "regions.p"
with open(regions, 'rb') as f:
    parking_boxes = pickle.load(f)

def parkingIoU(point, polygon_coordinates):
    point = Point(point)
    polygon = Polygon(polygon_coordinates)
    return point.within(polygon)

class ParkingSpace:
    _counter = 0  # Class variable to generate unique IDs

    def __init__(self, coords):
        # Increment the counter and assign a unique ID
        ParkingSpace._counter += 1
        self.id = ParkingSpace._counter

        # Set the provided coordinates
        self.vertices = np.array(coords,np.int32).tolist()
        polygon = Polygon(self.vertices)

        # Find the center of the polygon
        polygon_center = polygon.centroid.xy
        self.center_x, self.center_y = polygon_center[0][0], polygon_center[1][0]
        self.occupancy_stat = False
        self.preferred = False
        self.dis=0
        self.lat=0.0
        self.long=0.0

        # Set the parking space name as "p" followed by the counter
        self.name = f"p{self.id}"


psObjs=[]

for ps in parking_boxes:
    psObj = ParkingSpace(ps)
    psObjs.append(psObj)

#Updating GPS Coordinates
csv_file = 'Final GPS.csv'
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header if exists
    for row in csv_reader:
        for ps in psObjs:
            if row[0].replace(" ", "") == ps.name.replace(" ", ""):
                ps.lat = float(row[1])
                ps.long = float(row[2])
                ps.dis = float(row[3])
                break

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR image to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(frame_rgb)

    # Extract car bounding boxes
    car_boxes = results.xyxy[0].cpu().numpy()

    # Filter out boxes corresponding to cars (you may need to adjust class labels)
    car_boxes = car_boxes[car_boxes[:, 5] == 2]  # Assuming car class label is 2
    car_boxes = car_boxes[car_boxes[:, 4] >= 0.5]

    cxBig=[]
    cyBig=[]

    occupiedSpaces=[]

    # Calculatung center of car detection box
    for box in car_boxes:
        x_min, y_min, x_max, y_max, confidence, class_label = box
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        cx=int(x_min+x_max)//2
        cy=int(y_min+y_max)//2
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cxBig.append(cx)
        cyBig.append(cy)

    # Updating the status of parkingSpace using IoU
    for ps in psObjs:
        if len(cxBig) > 0 and len(cyBig) > 0:
            for i in range(len(cxBig)):
                point_inside=(cxBig[i],cyBig[i])
                polygon_coordinates=np.array(ps.vertices,np.int32)
                check_stat=parkingIoU(point_inside, polygon_coordinates)
                if check_stat==1:
                   ps.occupancy_stat=True
                   break
                else:
                    ps.occupancy_stat=False
    
    # Drawing lines on parking-spaces
    for ps in psObjs:
        yc=ps.center_y+15
        if ps.occupancy_stat==True:
            cv2.polylines(frame,[np.array(ps.vertices,np.int32)],True,(0,0,255),1)
            cv2.putText(frame,ps.name,(int(ps.center_x),int(yc)),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        else:
            cv2.polylines(frame,[np.array(ps.vertices,np.int32)],True,(0,255,0),1)
            cv2.putText(frame,ps.name,(int(ps.center_x),int(yc)),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    # Display the frame with bounding boxes
    cv2.imshow('RGB', frame)

    
    # Database Integration with no-sql
    json_objects = [json.dumps(space.__dict__) for space in psObjs]

    

    distinct_ids = collection.distinct("id")

    for data in json_objects:
            print(data)
            data_dict = json.loads(data)  # Convert JSON string to dictionary
            if data_dict["id"] in distinct_ids:
                update_result = collection.update_one(
                {"id": data_dict["id"]},
                {"$set": {"occupancy_stat": data_dict["occupancy_stat"]}})
            else:
                insert_result = collection.insert_one(data_dict)
                print("Inserted document ID:", insert_result.inserted_id)

    

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
client.close()
cv2.destroyAllWindows()
