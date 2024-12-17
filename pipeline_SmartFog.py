import sys
import cv2
import time
import argparse
import os
import math
from os import listdir
from os.path import isfile, join

import securityMOD as sMOD
from faceIdentify.detect import faceIdentity

ROOT_FACES_DB = "./faceIdentify/faces_database/"

outputPath = "output"


if __name__ == "__main__":

    faceIdentity = faceIdentity(ROOT_FACES_DB)

    vid = cv2.VideoCapture(0) 
     
    printLimit = 100
    n_frames = 0
    t_inf = 0.0

    while(True): 
        ret, frame = vid.read() 

        t0 = time.time()
        new_frame, detection = sMOD.inf_bodies_showing(frame, with_faces=True)
        if detection > 0: ident = faceIdentity.identify(frame)

        #faces = sMOD.inf_for_cropping(frame, with_faces=True)
        #ident = []
        #for face, name in faces: 
        #    ident.append(faceIdentity.identify(face))
        
        t_inf += time.time() - t0
        n_frames += 1

        #if len(ident) > 0:
        #    print(f"Person with weapon Identify: {ident}")

        if n_frames == printLimit:
            print(f"  FPS: {printLimit / t_inf:.2f} ")
            n_frames = 0
            t_inf = 0.0

        # Display the resulting frame 
        cv2.imshow('frame', frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  

    vid.release() 
    cv2.destroyAllWindows() 


'''
    if (opt.video == ""):
        #Process Image Batch
        files = get_files(opt)
        for idf, f in enumerate(files):
            frame = cv2.imread(f)

            #-----------------------------------------------------------
            # Inference Section
            #-----------------------------------------------------------
            t0 = time.time()
            frame = sMOD.inf_bodies_showing(frame, with_faces=opt.faces)
            t_inf = time.time() - t0

            print(f"[*] Process Image {idf+1}/{len(files)} '{f}' in {t_inf:.4f}(s)")

            if opt.show:
                cv2.imshow('frame', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cv2.imwrite(f"{outputPath}/img_out_{idf}.jpg", frame)
            #-----------------------------------------------------------
            #-----------------------------------------------------------

            #-----------------------------------------------------------
            # Only For Debug Purposes
            #-----------------------------------------------------------
            #crops = sMOD.inf_for_cropping(frame, with_faces=opt.faces)
            #for crop, name in crops:
            #    cv2.imshow('frame', crop)
            #    cv2.waitKey(0)
            #    cv2.destroyAllWindows()
            #-----------------------------------------------------------
            #-----------------------------------------------------------
            
            n_frames += 1
              
    else:
        #Process Video
        cap = cv2.VideoCapture(opt.video)
         
        outFile = outputPath + "/out-video.avi"
        frame_width  = int(cap.get(3))
        frame_height = int(cap.get(4))
    
        # Define the codec and create VideoWriter object.
        out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width, frame_height))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret : break
            
            frame = sMOD.inf_bodies_showing(frame, with_faces=opt.faces)
            out.write(frame)
            n_frames += 1

        out.release()
        cap.release()
'''
