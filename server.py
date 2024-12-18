import pickle
import socket
import struct
import cv2
import sys
import time
import connect
from   colors import ANSIColors

import threading
import queue

import securityMOD as sMOD

ALARMS   = int(sys.argv[1])
QMAXSIZE = int(sys.argv[2])

Q0 = queue.Queue(maxsize=QMAXSIZE)
Q1 = queue.Queue(maxsize=QMAXSIZE)

alarmsCount  = 0
displayAlarm = -1

#Alert Warning!
def warningManager(image, alarm):

    global alarmsCount
    global displayAlarm
    global ALARMS

    #Heuristic for display ALARM
    if alarmsCount < ALARMS and displayAlarm == -1: 
        if alarm: alarmsCount += 1
        else:     
            alarmsCount -= 1
            if alarmsCount < 0: alarmsCount = 0
        return image

    displayAlarm   += 1
    if displayAlarm == 10:
        displayAlarm = -1
        alarmsCount  = 0
        return image

    #Font configuration
    alarmsCount     = 0
    text            = "ALERTA 112"    #Message 
    text_color      = (255, 255, 255) #White
    rectangle_color = (0,     0, 255) #Red
    font            = cv2.FONT_HERSHEY_SIMPLEX
    font_scale      = 1
    font_thickness  = 2

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    x, y = 10, 30
    rect_x1, rect_y1 = x - 10, y - text_height - 10       # Box Left Top Corner
    rect_x2, rect_y2 = x + text_width + 10, y + baseline  # Box Right Bottom Corner

    #Draw Box
    cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), rectangle_color, thickness=-1)

    #Draw Text
    cv2.putText(image, text, (x, y), font, font_scale, text_color, font_thickness)

    return image

#Server connection TCP/IP
def serverDetector(Q0):
    
    port = int(sys.argv[3])
    ip   = '' #Listen from all ips

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, port))
    s.listen(10)
    print(f'SERVER LISTENING   : {ANSIColors.BG_YELLOW}WAITING{ANSIColors.RESET} ({ANSIColors.RED}0.0.0.0:{port}{ANSIColors.RESET})', end='\r', flush=True)

    conn, addr = s.accept()
    print(f'SERVER CONNECTED   : {ANSIColors.GREEN}OK{ANSIColors.RESET} (stablished from {ANSIColors.RED}{addr[0]}:{port}{ANSIColors.RESET})', end='\n', flush=True)

    data = b'' ### CHANGED
    payload_size = struct.calcsize("L") ### CHANGED

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

        #Put frame
        frame = pickle.loads(frame_data)
        if frame is not None:
            Q0.put(frame)

#Apply IA
def IADetector(Q0, Q1):
    while True:
        frame=Q0.get()
        if frame is not None:
            #If the queue is saturated, the frame is discarded.
            if Q1.qsize() < QMAXSIZE:
                frame, alarm = sMOD.inf_bodies_showing(frame, with_faces=False)
                Q1.put((frame, alarm))

#Display Frame Results
def displayDetector(Q1):
    nframes = 0

    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Display', int(sys.argv[4]), int(sys.argv[5]))

    t0 = time.time()
    while True:
        frame, alarm = Q1.get()
        if frame is not None:
            nframes += 1

            # Show Alert Warning (if need it)
            frame=warningManager(frame, alarm)

            #Display Frame
            cv2.imshow('Display', frame)
            cv2.waitKey(1)
            if nframes % 10 == 0:  
                tframes = time.time() - t0
                print(f"SERVER PERFORMANCE : {ANSIColors.BG_RED}{nframes/tframes:.2f}fps{ANSIColors.RESET}", end='\r')
                t0 = time.time()
                nframes = 0


if __name__ == "__main__":

    server  = threading.Thread(target=serverDetector,  args=(Q0,))
    IA      = threading.Thread(target=IADetector,      args=(Q0,Q1))
    display = threading.Thread(target=displayDetector, args=(Q1,))

    server.start()
    IA.start()
    display.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Server stopped.")

