# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import os
import sys
import numpy
import cv2
import math
from pathlib import Path
from ultralytics import YOLO
from PIL import Image,ImageDraw



# Modified from: https://docs.ultralytics.com/tasks/pose/#train YYang


def DoPerspectiveTransformationPoint(x,y):
    # Source points (e.g., corners of a quadrilateral in the original image)
    src_pts = numpy.float32([[2179, 2502], [3192, 2475], [3194, 1355], [2098, 1387]])

    # Destination points (e.g., corners of a rectangle in the desired output)
    dst_pts = numpy.float32([[2274, 2736], [3693, 2737], [3675, 682], [2250, 663]])

    # Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # A single point to transform
    point_to_transform = numpy.array([[x, y]], dtype=numpy.float32).reshape(-1, 1, 2)
    
    # Apply the perspective transformation
    transformed_point = cv2.perspectiveTransform(point_to_transform, M)
    
    # Access the transformed coordinates
    x_transformed = transformed_point[0][0][0]
    y_transformed = transformed_point[0][0][1]
    
    print(f"Original point: {point_to_transform[0][0]}")
    print(f"Transformed point: ({x_transformed}, {y_transformed})")

def DoPerspectiveTransformationPoints(points):
    # Source points (e.g., corners of a quadrilateral in the original image)
    src_pts = numpy.float32([[2179, 2502], [3192, 2475], [3194, 1355], [2098, 1387]])

    # Destination points (e.g., corners of a rectangle in the desired output)
    dst_pts = numpy.float32([[2274, 2736], [3693, 2737], [3675, 682], [2250, 663]])

    # Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # A single point to transform
    point_to_transform = numpy.array(points, dtype=numpy.float32).reshape(-1, 1, 2)
    
    # Apply the perspective transformation
    transformed_point = cv2.perspectiveTransform(point_to_transform, M)
    return transformed_point
 
def get_angle_with_x_axis(x0, y0, x1, y1):
  """
  Calculates the angle in degrees between a line and the positive x-axis.

  Args:
    x0: x-coordinate of the first point.
    y0: y-coordinate of the first point.
    x1: x-coordinate of the second point.
    y1: y-coordinate of the second point.

  Returns:
    The angle in degrees.
  """
  delta_y = y1 - y0
  delta_x = x1 - x0

  # Calculate the angle in radians using atan2
  angle_radians = math.atan2(delta_y, delta_x)

  # Convert the angle from radians to degrees
  angle_degrees = math.degrees(angle_radians)

  return angle_degrees

def get_angle_with_y_axis(x0, y0, x1, y1):
  """
  Calculates the angle in degrees between a line and the positive y-axis.

  Returns:
    The angle in degrees.
  """
  delta_y = abs(y1 - y0)
  delta_x = abs(x1 - x0)

  # Calculate the angle in radians using atan2
  angle_radians = math.atan(delta_x/delta_y) 

  # Convert the angle from radians to degrees
  angle_degrees = math.degrees(angle_radians)

  return angle_degrees

def draw_big_point(draw, center, radius, fill):
    x, y = center
    # Define the bounding box for the circle
    left_up = (x - radius, y - radius)
    right_down = (x + radius, y + radius)
    draw.ellipse([left_up, right_down], fill=fill)
    
def parse_roboflow_txt(file_path, image_width, image_height):
    polygons = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                coords = list(map(float, parts[1:]))  # skip class_id
                points = [(coords[i] * image_width, coords[i+1] * image_height) 
                          for i in range(0, len(coords), 2)]
    #            polygons.append(points)
                if len(points) >= 3:
                    polygons.append(points[0:4])   #only add 4 points
    except FileNotFoundError:
        return polygons
    return polygons

# (2.1) Train *********************************************

# Load a model
#model = YOLO("yolo11n-pose.pt")  # build a new model from YAML


# Train the model
#results = model.train(data="TillerAngles.yaml", epochs=500, imgsz=640,patience=200,batch=16)

#sys.exit()


# Load a model
model = YOLO("tilleranglev5.pt")  # load a custom model
#model = YOLO("seedingAll.pt")  # load an official model

#metrics = model.val(data="riceangles.yaml")  # no arguments needed, dataset and settings remembered
#metrics = model.val(data="TillerAngles.yaml")  # no arguments needed, dataset and settings remembered
#results = model.val(data="Seeding.yaml",save_txt=True)  # no arguments needed, dataset and settings remembered
#sys.exit()
# Predict with the model

image_path = "E:/Tiller Angles/train/images/East11_240618_DJI_0511_JPG.rf.99cc00c8b2d689cae140dabc75df0e5d.jpg"
file_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
results = model(image_path)  # predict on an image
#results = model("E:/Tiller Angles/DJI_0513.JPG",imgsz=1536)  # predict on an image
#results = model("M:/Samonte2024/2024-06-11/East11A45H45")  # predict on an image

#resultpath = Path("results")
resultpath = Path("results/2024-06-11")
#resultpath = Path("D:/BioSystemData/HarvestDataAnalysis/DigitalRiceSystem/TillerAngle/Predits")
resultpath = Path("D:/BioSystemData/HarvestDataAnalysis/DigitalRiceSystem/TillerAngle")
resultpath.mkdir(parents=True, exist_ok=True)

points = parse_roboflow_txt(file_path,6016,4008)
image = Image.open(image_path)
image_width, image_height = image.size
for xy in points:
    draw = ImageDraw.Draw(image)
    for a in xy:
        draw_big_point(draw,a,10,fill="red")
    draw.polygon(xy,fill =None, outline ="black",width=10)
image.save(os.path.join(resultpath, "East11_240618_DJI_0511_Obs.JPG"))   

for result in results:
    image = Image.open(image_path)
    image_width, image_height = image.size
    draw = ImageDraw.Draw(image)
    if result.keypoints is not None:
        xy = result.keypoints.xy  # x and y coordinates
        for ab in xy:
            for a in ab:
                draw_big_point(draw,a,10,fill="red")
            draw.polygon(ab.tolist(),fill =None, outline ="black",width=10)
    image.save(os.path.join(resultpath, "East11_240618_DJI_0511.JPG"))
sys.exit()

#results[0].show()  # Display results
#results[0].save()
#model.data = None
sys.exit()



# Access the results
for result in results:
    texts = []
    filename = f"{resultpath}/{Path(result.path).name}"  
    txtname = filename.replace(".jpg", ".txt").replace(".JPG", ".txt")
#    result.save(filename=filename)
#    result.save_txt(filename.replace(".JPG", ".txt"))
#    with open(txtname, "a", encoding="utf-8") as f:
    xy = result.keypoints.xy  # x and y coordinates
    for ab in xy:
#        angle1=get_angle_with_x_axis(ab[0][0],ab[0][1],ab[1][0],ab[1][1])
#        angle2=get_angle_with_x_axis(ab[2][0],ab[2][1],ab[3][0],ab[3][1])
#        angle = 180-abs(angle1)-abs(angle2)
        angleL=get_angle_with_y_axis(ab[0][0],ab[0][1],ab[1][0],ab[1][1])
        angleR=get_angle_with_y_axis(ab[3][0],ab[3][1],ab[2][0],ab[2][1])
#        print(angle)
        newpoints = DoPerspectiveTransformationPoints(ab.tolist())

        angleL0=get_angle_with_y_axis(newpoints[0][0][0],newpoints[0][0][1],newpoints[1][0][0],newpoints[1][0][1])
        angleR0=get_angle_with_y_axis(newpoints[3][0][0],newpoints[3][0][1],newpoints[2][0][0],newpoints[2][0][1])
#        angleAjust = 180-abs(angle1)-abs(angle2)
        line = ','.join([str(s) for s in ab.reshape(-1).tolist()])
        line1 = ','.join([str(s) for s in newpoints.reshape(-1).tolist()])
        texts.append(line+","+line1+","+str(angleL)+","+str(angleR)+","+str(angleL0)+","+str(angleR0))
#            f.writelines(line)
        
    with open(txtname, "w", encoding="utf-8") as f:
        f.writelines(text + "\n" for text in texts)
#    xyn = result.keypoints.xyn  # normalized
#    kpts = result.keypoints.data  # x, y, visibility (if available)

sys.exit()

#*********************************************************
# (1) Object Detection
#*********************************************************

# Load a pretrained YOLO11n model
#model = YOLO("yolo11n.pt")
model = YOLO("yolov8n.pt")


# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="coco8.yaml",  # Path to dataset configuration file
    epochs=20,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image

# Predict using a pretrained YOLO model (e.g., YOLO11n) on an image

results = model("https://ultralytics.com/images/bus.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model




#*********************************************************
# (2) Pose Estimation
#*********************************************************


# (2.1) Train *********************************************

# Load a model
#model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8-pose.yaml", epochs=20, imgsz=640)

# Train the model using a custom dataset
#results = model.train(data="your-dataset.yaml", epochs=20, imgsz=640)



# (2.2) Validate ******************************************

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
#model = YOLO("path/to/best.pt")  # load a custom model


# Validate the model
metrics = model.val(data="coco8-pose.yaml")  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category


# (2.3) Predict *******************************************

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
#model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
