import cv2
import numpy as np
import socket
import sys
import pickle
import struct
import connect
import time
from   colors import ANSIColors

import threading
import queue

QMAXSIZE = int(sys.argv[3])
Q = queue.Queue(maxsize=QMAXSIZE) 

def clientCamera(Q, cap):
    while True:
        if Q.qsize() < QMAXSIZE:
            ret,frame=cap.read()
            Q.put(frame)

def senderCamera(Q):
    ip   = sys.argv[1]
    port = int(sys.argv[2])
    clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clientsocket.connect((ip, port))

    print(f"CLIENT CONNECTION with [{ANSIColors.RED}{ip}:{port}{ANSIColors.RESET}]: {ANSIColors.GREEN}OK{ANSIColors.RESET}")

    while True:
        frame = Q.get()
        if frame is not None:
            # Serialize frame
            data = pickle.dumps(frame)
            # Send message length first
            message_size = struct.pack("L", len(data)) ### CHANGED
            # Then data
            clientsocket.sendall(message_size + data)

#By default this app capture from: PORT#0 - Webcam
if __name__ == "__main__":
    cap=cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(sys.argv[4]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(sys.argv[5]))

    client = threading.Thread(target=clientCamera, args=(Q,cap))
    sender = threading.Thread(target=senderCamera, args=(Q,))

    client.start()
    sender.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Client stopped.")
        cap.release()
        
