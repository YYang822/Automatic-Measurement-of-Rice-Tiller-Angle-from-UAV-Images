import os
import sys
import numpy as np
import cv2
import math
from pathlib import Path
from shapely.geometry import MultiPolygon,Polygon, LineString,Point,MultiPoint


def calculate_segmentation_metrics(ground_truth_mask, predicted_mask):
    """
    Calculates True Positives (TP) and False Positives (FP) for binary segmentation masks.

    Args:
        ground_truth_mask (np.ndarray): A 2D NumPy array representing the ground truth mask (binary).
        predicted_mask (np.ndarray): A 2D NumPy array representing the predicted mask (binary).

    Returns:
        tuple: A tuple containing:
            - tp (int): Number of True Positives.
            - fp (int): Number of False Positives.
            - fn (int): Number of False Negatives.
            - tn (int): Number of True Negatives.
    """
    # Ensure masks are boolean for logical operations
    ground_truth_mask = ground_truth_mask.astype(bool)
    predicted_mask = predicted_mask.astype(bool)

    # True Positives (TP): Pixels correctly identified as part of the object
    tp = np.sum(predicted_mask & ground_truth_mask)

    # False Positives (FP): Pixels incorrectly identified as part of the object
    fp = np.sum(predicted_mask & ~ground_truth_mask)

    # False Negatives (FN): Pixels that are part of the object but were missed by the prediction
    fn = np.sum(~predicted_mask & ground_truth_mask)

    # True Negatives (TN): Pixels correctly identified as background
    tn = np.sum(~predicted_mask & ~ground_truth_mask)

    return tp, fp, fn, tn

        
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


def parse_roboflow_txt_Transform(file_path, image_width, image_height,M):
    polygons = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                coords = list(map(float, parts[1:]))  # skip class_id
                points = [(coords[i] * image_width, coords[i+1] * image_height) 
                          for i in range(0, len(coords), 2)]
                points =DoPerspectiveTransformationPoints(points,M)
    #            polygons.append(points)
                if len(points) >= 3:
                    polygons.append(points)
    except FileNotFoundError:
        return polygons
    return polygons

def parse_seg_txt(file_path, image_width, image_height):
    polygons = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                coords = list(map(float, parts[1:]))  # skip class_id
                points = [(coords[i] * image_width, coords[i+1] * image_height) 
                          for i in range(0, len(coords), 2)]
                polygons.append(points)
    #            if len(points) >= 3:
#                polygons.append(Polygon(points))
    except FileNotFoundError:
        return polygons
    return polygons


def compute_iou(points1, points2):
    poly1 = Polygon(points1)
    poly2 = Polygon(points2)
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union != 0 else 0

def compute_area(gt_polygons):
    area=0
    for j, gt in enumerate(gt_polygons):
        area += int(Polygon(gt).area)
    return area
    
def evaluate_segmentation(gt_polygons, pred_polygons, iou_threshold=0.50):
    
    matched_gt = set()
    matched_pred = set()
    gtArea = compute_area(gt_polygons)
    preArea = compute_area(pred_polygons)
    if len(gt_polygons)>0 and len(pred_polygons)>0 :
        for i, pred in enumerate(pred_polygons):
            for j, gt in enumerate(gt_polygons):
                if j in matched_gt:
                    continue
                iou = compute_iou(pred, gt)
                if iou >= iou_threshold:
                    matched_gt.add(j)
                    matched_pred.add(i)
                    break
    
        tp = len(matched_pred)
        fp = len(pred_polygons) - tp
        fn = len(gt_polygons) - tp
    else:
        if len(gt_polygons)>0:
            tp=0
            fp=0
            fn=len(gt_polygons)
        else:
            tp =0
            fp =len(pred_polygons)
            fn=0
    if tp+fp>0:
        precision = str(tp/(tp+fp))
    else:
        precision=""
    if tp+fn>0:
        recall = str(tp/(tp+fn))
    else:
        recall = ""

    if len(gt_polygons)>0:
        return str(tp)+","+str(fp)+","+str(fn)+","+precision+","+recall+","+str(gtArea)+","+str(preArea)
    else: 
        return ""



def DoPerspectiveTransformationPoint(x,y,M):
    #reference 45 degree and 4.5 meter hight
    # Source points (e.g., corners of a quadrilateral in the original image)
#    src_pts = np.float32([[2179, 2502], [3192, 2475], [3194, 1355], [2098, 1387]])
#    src_pts = np.float32([[1618, 2610], [5076, 2566], [5456, 714], [1546,679]])
    
    # Destination points (e.g., corners of a rectangle in the desired output)
#    dst_pts = np.float32([[2274, 2736], [3693, 2737], [3675, 682], [2250, 663]])
#    dst_pts = np.float32([[1122,3134], [5804,3277], [5870,118], [1167,77]])

    # Calculate the perspective transformation matrix
#    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # A single point to transform
    point_to_transform = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
    
    # Apply the perspective transformation
    transformed_point = cv2.perspectiveTransform(point_to_transform, M)
    
    # Access the transformed coordinates
    x_transformed = transformed_point[0][0][0]
    y_transformed = transformed_point[0][0][1]
    
    print(f"Original point: {point_to_transform[0][0]}")
    print(f"Transformed point: ({x_transformed}, {y_transformed})")

def DoPerspectiveTransformationPoints(points,M):
    # Source points (e.g., corners of a quadrilateral in the original image)
#    src_pts = np.float32([[2179, 2502], [3192, 2475], [3194, 1355], [2098, 1387]])
#    src_pts = np.float32([[1618, 2610], [5076, 2566], [5456, 714], [1546,679]])
    
    # Destination points (e.g., corners of a rectangle in the desired output)
#    dst_pts = np.float32([[2274, 2736], [3693, 2737], [3675, 682], [2250, 663]])
#    dst_pts = np.float32([[1122,3134], [5804,3277], [5870,118], [1167,77]])
    # Calculate the perspective transformation matrix
#    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # A single point to transform
    point_to_transform = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    
    # Apply the perspective transformation
    transformed_point = cv2.perspectiveTransform(point_to_transform, M)
    Jpoints = [(point[0][0], point[0][1]) for point in transformed_point]
    return Jpoints

def DoPerspectiveTansformationImage(img,M):
    # Source points (e.g., corners of a quadrilateral in the original image)
#    src_pts = np.float32([[2179, 2502], [3192, 2475], [3194, 1355], [2098, 1387]])
#    src_pts = np.float32([[1618, 2610], [5076, 2566], [5456, 714], [1546,679]])
    
    # Destination points (e.g., corners of a rectangle in the desired output)
#    dst_pts = np.float32([[2274, 2736], [3693, 2737], [3675, 682], [2250, 663]])
#    dst_pts = np.float32([[1122,3134], [5804,3277], [5870,118], [1167,77]])

    rows, cols,z = img.shape
    # Calculate the perspective transformation matrix
#    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    reflected_img = cv2.warpPerspective(img, M,(int(cols),int(rows)))
    return reflected_img
 
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

def findintersect_midPoints(poly, line):
    point=Point(-1,-1)
    intersection = poly.intersection(line)
    if isinstance(intersection, MultiPoint):
    # Extract the coordinates of the intersection points
        points = [(point.x, point.y) for point in intersection.geoms]
#        print("points length="+str(len(points)))
        point = Point((points[0][0]+points[1][0])/2,(points[0][1]+points[1][1])/2)

    elif intersection.is_empty:
#        print("No intersection")
        return None
    else:
        try:
            point = intersection.interpolate(0.5, normalized=True)
        except:
#            print("error")
            return None

#        print("this is linestring")
#        print(point)
    if point.within(poly):
        return point
    else:
        return None

def get_longest_side_midpoints_curve(polygon_coords,width,length):

    multi_polygon =Polygon(polygon_coords)
    if not multi_polygon.is_valid:
        multi_polygon = multi_polygon.buffer(0) # Attempt to fix the polygon
        
    area = multi_polygon.area

    x_coords = [p[0] for p in polygon_coords]
    y_coords = [p[1] for p in polygon_coords]
    
    min_x = min(x_coords)
    max_x = max(x_coords)
    x_length = (max_x - min_x) /width
    
    min_y = min(y_coords)
    max_y = max(y_coords)
    y_length = (max_y - min_y)/length
    
    resultstr = str(max_x)+","+str(min_x)+","+str(max_y)+","+str(min_y)+","+str(area)
    
    numP = 11
    middle_points = []
    if y_length/x_length>= 1:
        delta = (max_y - min_y)/numP
        for x in range(1, numP):
            line = LineString([(min_x-10, min_y+x*delta), ( max_x+10, min_y+x*delta)])
            point = findintersect_midPoints(multi_polygon,line) 
            if point is not None:
                middle_points.append((point.x,point.y))
    else:
        delta = (max_x - min_x)/numP
        for x in range(1, numP):
            line = LineString([(min_x+x*delta,min_y-10), ( min_x+x*delta,max_y+10)])
            point = findintersect_midPoints(multi_polygon,line) 
            if point is not None:
                middle_points.append((point.x,point.y))
    
   
    delimiter = ";"
    decimal_places =2
    resultstr =resultstr+","+delimiter.join(f"({point[0]:.{decimal_places}f} {point[1]:.{decimal_places}f})" for point in middle_points)
#    print(resultstr)
    return resultstr, middle_points
 
 
def calculate_curvature(p1, p2, p3):
    """
    Calculates the Menger curvature for three 2D points.
    Curvature is the reciprocal of the radius of the circle passing through the points.
    """
    # Convert points to numpy arrays for easier calculations
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Calculate distances between points (sides of the triangle)
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p2 - p3)
    c = np.linalg.norm(p3 - p1)

    # Calculate the area of the triangle using Heron's formula
    s = (a + b + c) / 2.0 # Semi-perimeter
    # Use a safe version of Heron's formula to avoid potential floating point issues for very flat triangles
    if s * (s - a) * (s - b) * (s - c)<0:
      if abs((p1[1] - p2[1]) * (p1[0] - p3[0])-(p1[1] - p3[1]) * (p1[0] - p2[0]))<0.000000001:
          return 0
    k = np.sqrt(s * (s - a) * (s - b) * (s - c))
    # Curvature formula: kappa = 4*k / (a*b*c)
    if a * b * c == 0.0:
        return 0.0 # Collinear points or identical points
    else:
        return 4.0 * k / (a * b * c)

def average_curvatures_rolling(points, window_size=3):
    """
    Calculates the moving average of curvatures for a list of points
    using a specified window size.
    """
    curvatures = []
    # Curvature can only be calculated starting from the 2nd point (requires p1, p2, p3)
    for i in range(1, len(points) - 1):
        p1 = points[i-1]
        p2 = points[i]
        p3 = points[i+1]
        curv = calculate_curvature(p1, p2, p3)
        if str(curv)!="nan":
            curvatures.append(curv)
    
    # Calculate the average of all computed curvatures
    if not curvatures:
        return 0.0
    if str(np.mean(curvatures))=="nan":
        print("error")
    return np.mean(curvatures)

def calculate_curvature1(x_coords, y_coords):
    """
    Calculates the curvature of a 2D curve defined by x and y coordinates.

    Args:
        x_coords (np.array): Array of x-coordinates.
        y_coords (np.array): Array of y-coordinates.

    Returns:
        np.array: Array of curvature values for each point.
    """
    if len(x_coords) != len(y_coords):
        raise ValueError("x_coords and y_coords must have the same length.")
    if len(x_coords) < 3:
        # Curvature requires at least 3 points for a meaningful calculation
        return np.zeros_like(x_coords) 

    # Calculate first derivatives using numpy.gradient
    dx = np.gradient(x_coords)
    dy = np.gradient(y_coords)

    # Calculate second derivatives
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Calculate curvature using the formula
    # K = |d2y/dx2| / (1 + (dy/dx)^2)^(3/2)
    # This can be rewritten using the chain rule for derivatives with respect to a parameter
    # Curvature = |dx * d2y - d2x * dy| / (dx**2 + dy**2)**1.5
    curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**1.5
    
    return curvature

def doTillerAngle(result_path,output_path,image_width,image_height):
    for root, dirs, files in os.walk(result_path): 
           for name in files:
               if name.endswith('.txt'):
                   file_path = os.path.join(root, name)

                   txtname = os.path.join(output_path,name)
                   results = parse_roboflow_txt(file_path,image_width,image_height)
                   texts = []
                   for ab in results:
#                        angle1=get_angle_with_x_axis(ab[0][0],ab[0][1],ab[1][0],ab[1][1])
#                        angle2=get_angle_with_x_axis(ab[2][0],ab[2][1],ab[3][0],ab[3][1])
#                        angle = 180-abs(angle1)-abs(angle2)
                        angleL=get_angle_with_y_axis(ab[0][0],ab[0][1],ab[1][0],ab[1][1])
                        angleR=get_angle_with_y_axis(ab[3][0],ab[3][1],ab[2][0],ab[2][1])
                #        print(angle)
                        newpoints = DoPerspectiveTransformationPoints(ab)
            
                        angleL0=get_angle_with_y_axis(newpoints[0][0],newpoints[0][1],newpoints[1][0],newpoints[1][1])
                        angleR0=get_angle_with_y_axis(newpoints[3][0],newpoints[3][1],newpoints[2][0],newpoints[2][1])
#                        angleAjust = 180-abs(angle1)-abs(angle2)
                        line = ','.join([str(s) for s in ab])
                        line1 = ','.join([str(s) for s in newpoints])
                        line = line.replace("(", "").replace(")", "")
                        line1 = line1.replace("np.float32", "").replace("(", "").replace(")", "")
                        texts.append(line+","+line1+","+str(abs(angleL))+","+str(abs(angleR))+","+str(abs(angleL0))+","+str(abs(angleR0)))
                #            f.writelines(line)
                        
                   with open(txtname, "w", encoding="utf-8") as f:
                        f.writelines(text + "\n" for text in texts)

#points =[(1470.00,800.27),(1470.00,800.55),(1470.00,800.82),(1470.00,801.09),(1470.00,801.36),(1470.00,801.64),(1470.00,801.91),(1470.00,802.18),(1470.00,802.45),(1470.00,802.73)]

#curve = average_curvatures_rolling(points)
#curev2 = calculate_curvature1(x_coords,y_coords)
#print("curev="+str(curve))

#image_width = 1504
#image_height =1002
image_width = 6016
image_height =4008
Xfactor = image_width/6016
Yfactor = image_height/4008
src_pts = np.float32([[1618, 2610], [5076, 2566], [5456, 714], [1546,679]])
dst_pts = np.float32([[1122,3134], [5804,3277], [5870,118], [1167,77]])
for p in src_pts:
    p[0] = p[0]*Xfactor
    p[1] = p[1]*Yfactor
for p in dst_pts:
    p[0] = p[0]*Xfactor
    p[1] = p[1]*Yfactor

M = cv2.getPerspectiveTransform(src_pts, dst_pts)

#p1=DoPerspectiveTransformationPoint(0, 0,M)
#p2=DoPerspectiveTransformationPoint(0, 1002,M)
#p3=DoPerspectiveTransformationPoint(1504, 1002,M)
#p4=DoPerspectiveTransformationPoint(1504, 0,M)
#sys.exit()
# Example usage


#img = cv2.imread('E:/Tiller Angles/test/images/East_220708_DJI_0270_JPG.rf.c0c4fbc10bf3be77d7e88c9d16b95372.jpg')
#reflected_img = DoPerspectiveTansformationImage(img)
#cv2.imshow("Original Image", img)
#cv2.imshow("new imag",reflected_img)
#cv2.imwrite('D:/BioSystemData/HarvestDataAnalysis/DigitalRiceSystem/TillerAngle/Labels/reflection.jpg', reflected_img)

output_path ="D:/BioSystemData/HarvestDataAnalysis/DigitalRiceSystem/TillerAngle/Labels"
result_path = "E:/Tiller Angles"
#doTillerAngle(result_path, output_path, image_width, image_height)
#sys.exit()

curveb=True
directory_path = "E:/Plot Segmentation.v1i.yolov11/train/labels"  # Replace with your directory path
#directory_path = "D:/YOLO/V11/ultralytics-main/Digital Crops/results/PaniclePredict"
Rfilename = os.path.join("E:/Panicles", "resultsAreasTransf.txt")  #for plot segmentation area
#Rfilename = os.path.join("E:/Panicles", "resultsTAMUTransf.txt")

predict_path=Path("D:/YOLO/V11/ultralytics-main/Digital Crops/results/PanicleTextTAMU")
#predict_path = Path("E:/Panicles/txt")
result_path=Path("E:/Panicles/Predict_Transf")
result_path.mkdir(parents=True, exist_ok=True)


#img = cv2.imread('E:/DJI_0010.JPG')
#reflected_img = DoPerspectiveTaansformationImage(img,M)
#cv2.imshow("Original Image", img)
#cv2.imshow("new imag",reflected_img)
#cv2.imwrite('E:/reflection_DJI_0010.jpg', reflected_img)
#sys.exit()

doTransForm = True
dir_list = os.listdir(result_path)
result = []
for root, dirs, files in os.walk(directory_path): 
       for name in files:
#           if name.endswith('.csv') and Path(root).name+"_"+name not in dir_list:
               file_path = os.path.join(root, name)
#               file_path_P = os.path.join(root, name.replace(".txt", ".csv"))
               file_path_P = os.path.join(predict_path, name.replace(".txt", ".csv"))
#               txtname = os.path.join(result_path,name)
               txtname = os.path.join(result_path,Path(root).name+"_"+name.replace(".txt", ".csv"))
               if doTransForm:
                   gt_polygons = parse_roboflow_txt_Transform(file_path,image_width,image_height,M)
               else:
                  gt_polygons = parse_roboflow_txt(file_path,image_width,image_height)
               area=compute_area(gt_polygons)
               result.append(name.replace(".txt",".jpg")+","+str(area))
               continue
               if doTransForm:
                   pred_polygons = parse_roboflow_txt_Transform(file_path_P,image_width,image_height,M)
               else :
                   pred_polygons = parse_roboflow_txt(file_path_P,image_width,image_height)
               texts = []
               if curveb:
                   for points in pred_polygons:
           #            ajpoints = DoPerspectiveTransformationPoints(points)
                       resultstr,curve_points = get_longest_side_midpoints_curve(points,image_width,image_height)
                       curvatures = average_curvatures_rolling(curve_points)
                       if str(curvatures)=="nan":
                           print("error")
                       texts.append(resultstr+","+str(curvatures))
                   if os.path.exists(txtname): # Check if the file exists
                       os.remove(txtname)
                   with open(txtname, "w", encoding="utf-8") as f:
                       f.writelines(text + "\n" for text in texts)
#               tpstring = evaluate_segmentation(gt_polygons, pred_polygons)
#               result.append(Path(file_path ).name.replace(".txt",".jpg")+","+tpstring)
#sys.exit()

if os.path.exists(Rfilename): # Check if the file exists
    os.remove(Rfilename)
with open(Rfilename, "w", encoding="utf-8") as f:
    f.writelines(text + "\n" for text in result)
sys.exit()

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
#    file_path_P = file_path.replace("labels", "txt")
    file_path_P = os.path.join(predict_path, filename.replace(".txt", ".csv"))
    texts = []
    txtname = file_path.replace("labels", "Results")
    if os.path.isfile(file_path):
        gt_polygons = parse_roboflow_txt(file_path,image_width,image_height)
        pred_polygons = parse_roboflow_txt(file_path_P,image_width,image_height)
        if curveb:
            for points in pred_polygons:
    #            ajpoints = DoPerspectiveTransformationPoints(points)
                resultstr,curve_points = get_longest_side_midpoints_curve(points,image_width,image_height)
                curvatures = average_curvatures_rolling(curve_points)
                if str(curvatures)=="nan":
                    print("error")
                texts.append(resultstr+","+str(curvatures))
    
            with open(txtname, "w", encoding="utf-8") as f:
                f.writelines(text + "\n" for text in texts)
        tpstring = evaluate_segmentation(gt_polygons, pred_polygons)
        result.append(Path(file_path ).name.replace(".txt",".jpg")+","+tpstring)

with open(Rfilename, "w", encoding="utf-8") as f:
    f.writelines(text + "\n" for text in result)
