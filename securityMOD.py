import sys
import cv2
import time
import argparse
import os
import math
from os import listdir
from os.path import isfile, join

from ultralytics import YOLO
import face_recognition

#Tensor Options
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# COCO yalm: Object Classes
#classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#              "teddy bear", "hair drier", "toothbrush" ]

#For Coco
classNames   = ["person"]

#For Weapons Detection
weaponsNames = ["Pistol", "SmartPhone", "Knife", "Wallet", "Ticket", "Card", "Weapon"]

#Color for boxes
red_color = (0, 0, 200)
rcolors   = [ (0,   100, 230),
              (100, 0,   230),
              (230, 200, 0),
              (200, 200, 100),
              (0,   230, 100),
              (200, 230, 0),
              (0,   230, 0),
              (100, 230, 0),
              (0,   230, 200),
              (0,   0,   230),
              (200, 0,   230),
              (200, 0,   100),
              (230, 100, 0),
              (200, 50,  100),
              (200, 100, 100),
              (100, 0,   0) ]
num_colors = len(rcolors)


print("\nLoading models...", end= " ")

t0 = time.time()

#Load Models
Body   = YOLO("models/body_yolov8n.pt")
Weapon = YOLO("models/weapon_yolov8n.pt")
Face   = YOLO("models/yolov8n-face.pt")

#Warning: Force to Load weight models into RAM with inference 
Body  ("models/.force_off_loading.jpg", conf=0.2, verbose=False)
Weapon("models/.force_off_loading.jpg", conf=0.2, verbose=False)
Face  ("models/.force_off_loading.jpg", conf=0.2, verbose=False)

t_load = time.time() - t0

print(f"Done in {t_load:.2f}(s)!\n", end= "\n")

def face_crop(faces, name, img):
    faces_cropped = []
    for face in faces:
        x1, y1, x2, y2 = face.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
        crop = img[y1:y2, x1:x2]
        faces_cropped.append([crop, name])

    return faces_cropped

def bodies_crop(pairs, objects, bodies, img):
    final_bodies = []
    bodies_cropped = []
    for pair_obj, pair_body in pairs:
        if pair_body not in bodies_cropped:
            body = bodies[pair_body]
            obj  = objects[pair_obj]
            x1, y1, x2, y2 = body.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            crop = img[y1:y2, x1:x2]
            final_bodies.append([crop, weaponsNames[int(obj.cls[0])]])
            bodies_cropped.append(pair_body)

    return final_bodies

def get_center(box):
    x0, y0, x1, y1 = box
    width  = x1 - x0
    height = y1 - y0
    return (x0 + int(width/2), y0 + int(height/2))

def calculate_distance(target0, target1):
    center0 = get_center(target0)
    center1 = get_center(target1)
    diff = (center1[0] - center0[0], center1[1] - center0[1])
    return math.sqrt(diff[0]*diff[0] + diff[1]*diff[1])

def pairing_object_to_bodies(objects, bodies):
    pairs = []
    for o, obj in enumerate(objects):
        min_distance, body_pair = sys.maxsize, -1
        for b, body in enumerate(bodies):
            distance = calculate_distance(body.xyxy[0], obj.xyxy[0])
            if distance < min_distance:
                body_pair, min_distance = b, distance
        if body_pair != -1:
            pairs.append([o, body_pair])

    return pairs

def draw_object_box(img, box, tag, pair_color):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

    cv2.rectangle(img, (x1, y1), (x2, y2), pair_color, 2)
    conf  = math.ceil((box.conf[0]*100))/100
    label = f'{tag} {conf:.2f}'

    org = [x1, y1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    color = (255, 255, 255)
    thickness = 2

    cv2.putText(img, label, org, font, fontScale, color, thickness)

    return img

def draw_pair_results(img, pairs, objects, bodies):
    bodies_found = {}
    color_id = 0
    for obj_id, body_id in pairs:
        body = bodies[body_id]
        obj  = objects[obj_id]
        if body_id not in bodies_found:
            pair_color = rcolors[color_id]
            draw_object_box(img, body, "Person", pair_color)
            
            bodies_found[body_id] = pair_color
            color_id += 1
            if color_id == num_colors: color_id = 0
        else:
            pair_color = bodies_found[body_id]

        draw_object_box(img, obj, weaponsNames[int(obj.cls[0])], pair_color)

    for o_id, obj in enumerate(objects):
        found = False
        for pairObj_id, pairBody_id in pairs:
            if o_id == pairObj_id:
                found = True
                break
        if not found:
            draw_object_box(img, obj, weaponsNames[int(obj.cls[0])], red_color)

    for b_id, body in enumerate(bodies):
        found = False
        for pairObj_id, pairBody_id in pairs:
            if b_id == pairBody_id:
                found = True
                break
        if not found:
            draw_object_box(img, body, "Person", red_color)

    return img

def draw_results(img, boxes):
    for b_id, box in enumerate(boxes):
        draw_object_box(img, box, "Face", red_color)

    return img


#--------------------------------------------------------------------------------------
# INFERENCE
#--------------------------------------------------------------------------------------

def inf_bodies_showing(img, with_faces=False):
    #YOLOv8 Inference
    bodies  = Body(img, classes=0, verbose=False)[0].boxes  #Warning! Only for one image.
    objects = Weapon(img, conf=0.3, verbose=False)[0].boxes
    faces   = []

    if with_faces:
        faces = Face(img, classes=0, verbose=False)[0].boxes
    
    #Pairing Weapons with Bodies
    pairs   = pairing_object_to_bodies(objects, bodies) 
    
    #Draw results 
    img    = draw_pair_results(img, pairs, objects, bodies)
    
    if with_faces: img = draw_results(img, faces)

    return img, len(pairs)

def inf_for_cropping(img, with_faces=False):
    #YOLOv8 Inference
    bodies  = Body(img, classes=0, verbose=False)[0].boxes  #Warning! Only for one image.
    objects = Weapon(img, conf=0.3, verbose=False)[0].boxes
    faces   = Face(img, classes=0, verbose=False)[0].boxes
    
    #Pairing Weapons with Bodies
    pairs   = pairing_object_to_bodies(objects, bodies) 
    
    bodies_cropped = bodies_crop(pairs, objects, bodies, img)

    if with_faces:
        faces_cropped = []
        for body, name in bodies_cropped:
            faces = Face(body, classes=0, verbose=False)[0].boxes
            faces_cropped = faces_cropped + face_crop(faces, name, body)
        return faces_cropped 
    else:
        return bodies_cropped



