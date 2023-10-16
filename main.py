import cv2
import random
import numpy as np

thres = 0.5 # Threshold to detect object
#img = cv2.imread("assets/1.jpg")
cap = cv2.VideoCapture("assets/12345.mp4")
cap.set(3,640)
cap.set(4,480)

color = list(np.random.random(size=3) * 256)
#colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for j in range(10)]
classNames = []
classFile = 'assets/coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'assets/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "assets/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=0.5)
    print(classIds,bbox)

    if len(classIds) !=0:
        for classIds, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color,4)
            #cv2.rectangle(img,box,(colors[classIds % len(colors)]),3)
            cv2.putText(img,classNames[classIds-1].upper(),(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

            cv2.putText(img, str(round(confidence*100,2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("output",img)
    if cv2.waitKey(1) & 0xFF ==ord('a'):
        break
