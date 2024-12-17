import cv2
import numpy as np
import socket
import sys
import pickle
import struct
import connect
import time

import threading
import queue

QMAXSIZE = 10

Q = queue.Queue(maxsize=QMAXSIZE) 

def camera(Q):
    cap=cv2.VideoCapture(0)

    if cap:
        print("Client Camera Configuration: OK")
    else:
        print("Client Camera Configuration: ERROR")
        sys.exit(-1)

    while True:
        if Q.qsize() < QMAXSIZE:
            ret,frame=cap.read()
            Q.put(frame)

def sender(Q):
    ip, port = connect.get_ip_port()
    clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clientsocket.connect((ip, port))

    print(f"Client Connection [{ip}:{port}]: OK")

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
    producer = threading.Thread(target=camera, args=(Q,))
    consumer = threading.Thread(target=sender, args=(Q,))

    producer.start()
    consumer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Client stopped.")

