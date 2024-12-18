import sys
import cv2
import time
import argparse
import os
import math
from os import listdir
from os.path import isfile, join

import securityMOD as sMOD

outputPath = "output"

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgPath', '-p', type=str,            default="./img/", help='Images input path')
    parser.add_argument('--video',   '-e', type=str,            default="",       help='Video  input file name')
    parser.add_argument('--faces',   '-f', action='store_true', default=False,    help="Face detection active.")
    parser.add_argument('--show',    '-s', action='store_true', default=False,    help="Show frames processed.")
    opt = parser.parse_args()
    return opt


def sortFiles(e):
    return int(e.split("/")[2].split(".")[0][1:])


def get_files(opt):
    return  [join(opt.imgPath, f) for f in listdir(opt.imgPath) if isfile(join(opt.imgPath, f))]


if __name__ == "__main__":
    opt   = parse_options()
   
    n_frames = 0

    if (opt.video == ""):
        #Process Image Batch
        files = get_files(opt)
        for idf, f in enumerate(files):
            frame = cv2.imread(f)

            #-----------------------------------------------------------
            # Inference Section
            #-----------------------------------------------------------
            t0 = time.time()
            frame,alarm = sMOD.inf_bodies_showing(frame, with_faces=opt.faces)
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
            
            frame,alarm = sMOD.inf_bodies_showing(frame, with_faces=opt.faces)
            out.write(frame)
            n_frames += 1

        out.release()
        cap.release()


    print("")
    print("==========================================================")
    print("|                  INFERENCE INFORMATION                 |")
    print("==========================================================")
    print(f"  [*] Total Images processed : {n_frames} ")
    print("==========================================================")

