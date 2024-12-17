import pickle
import socket
import struct
import cv2
import sys
import time
import connect

#import argparse
#import os
#from os import listdir
#from os.path import isfile, join
#from objectDetection.detect import YoloV5OD, pairing_object_to_bodies, save_pair_results, body_crop, face_crop

HOST = ''
PORT = 9009

ROOT_WEIGHTS  = "./objectDetection/yoloV5-weights/"
ROOT_FACES_DB = "./faceIdentify/faces_database/"

WEIGHTS = [ROOT_WEIGHTS+"yolov5s.pt",
           ROOT_WEIGHTS+"weapons-v5s-new.pt",           
           ROOT_WEIGHTS+"face_detection_yolov5s.pt"]

#WEIGHTS_ALL = ROOT_WEIGHTS + "bodies-weapons-300epc.pt"

WEAPONS_OUTPUT = "weapons-detected"
CROPS_OUTPUT   = "bodies-croped"
FACES_OUTPUT   = "faces-detected"

#def parse_options():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--imgPath',    '-p', type=str,            default="./data", help='Images input path')
#    parser.add_argument('--img',        '-i', type=str,            default="",       help='Image  input file name')
#    parser.add_argument('--outputPath', '-o', type=str,            default="./run",  help='Image processing output path')
#    parser.add_argument('--video',      '-e', type=str,            default="",       help='Video  input file name')
#    parser.add_argument('--faces',      '-f', action='store_true', default=False,    help="Face detection active.")
#    parser.add_argument('--verbose',    '-v', action='store_true', default=False,    help="Information about the inference.")
#    parser.add_argument('--webcam',     '-w', action='store_true', default=False,    help='Video from webcam/Server')
#    opt = parser.parse_args()
#    
#    return opt

#def sortFiles(e):
#    return int(e.split("/")[2].split(".")[0][1:])

#def prepare_dirs(opt):
#    outWPath = join(opt.outputPath, WEAPONS_OUTPUT)
#    if opt.faces:
#        outCPath = join(opt.outputPath, FACES_OUTPUT)
#    else:
#        outCPath = join(opt.outputPath, CROPS_OUTPUT)
#
#    if opt.video == "":
#        os.mkdir(outWPath)    
#        os.mkdir(outCPath)
#    
#    return (outWPath, outCPath)

#def get_files(opt):
#    files = []
#    if opt.img != "":
#        files.append(opt.img)
#    elif opt.video != "":
#        files.append(opt.video)
#    else:
#        files = [join(opt.imgPath, f) for f in listdir(opt.imgPath) if isfile(join(opt.imgPath, f))]
#    
#    #if opt.video == "":
#    #    files.sort(key=sortFiles)
#
#    return files

#def inference_pipeline(img):
#    t0 = time.time()
#    
#    #Weapons and Bodies Inferece. Body and Weapons detection
#    #weapons_bodies = weaponsBodiesOD.do_inference(img)
#    bodiesKnifes  = bodyOD.do_inference(img, class_filter=['person', 'knife'])
#    weapons       = weaponOD.do_inference(img)
#    bodies        = []
#    for obj in bodiesKnifes:
#        bodies.append(obj) if obj.name == 'person' else weapons.append(obj) 
#     
#    #Pairing bodies with weapons and cropping bodies
#    pairs       = pairing_object_to_bodies(weapons, bodies) 
#    bodies_crop = body_crop(pairs, weapons, bodies, img)
#
#    identities=[]
#    #For each body crop the face and identify the face
#    for i, b in enumerate(bodies_crop):
#        faces = faceOD.do_inference(b[0])
#        faces_crop = face_crop(faces, b[0])
#        for j, face in enumerate(faces_crop):
#            ident = faceIdentity.identify(face)
#            if len(ident) > 0 : identities.append(ident[0])  
#            
#    #End All pipeline Inference
#    inf_time = time.time() - t0
#
#    return pairs, weapons, bodies, identities, inf_time 




if __name__ == "__main__":

    #bodyOD   = YoloV5OD(WEIGHTS[0], conf_thres=0.3)
    #weaponOD = YoloV5OD(WEIGHTS[1], conf_thres=0.4)
    #faceOD   = YoloV5OD(WEIGHTS[2], conf_thres=0.3)


    ip, port = connect.get_ip_port()
    ip='' #Listen from all ips

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((ip, port))
    print('Socket bind complete')
    
    s.listen(10)
    print('Socket now listening')

    conn, addr = s.accept()

    data = b'' ### CHANGED
    payload_size = struct.calcsize("L") ### CHANGED

    t0 = time.time()
    nframes = 0

    while True:

        # Retrieve message size
        while len(data) < payload_size:
            data += conn.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

        # Retrieve all data based on message size
        while len(data) < msg_size:
            data += conn.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)

        #Detection
        #pairs, weapons, bodies, identities, inf_time = inference_pipeline(frame)
        #idents_str = parse_identities_text(identities, pairs)
        #save_pair_results(pairs, weapons, bodies, frame, idents_str, display="frame")
        
        nframes += 1

        # Display
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        if nframes % 10 == 0:  

            tframes = time.time() - t0

            print(f"FPS: {nframes/tframes:.2f}", end='\r')

            t0 = time.time()
            nframes = 0

