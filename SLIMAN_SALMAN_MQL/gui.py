




from pathlib import Path
import cv2
import numpy as np
import imutils
from tkinter import filedialog
from tkinter import messagebox
import math
import cv2 as cv
import Person
import time
import datetime
import cv2
import logging as log
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker
from itertools import combinations
import math
import time
import pyttsx3
from tkinter import *
engine = pyttsx3.init()



# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
def picture():
    

    protopath = "MobileNetSSD_deploy.prototxt"
    modelpath = "MobileNetSSD_deploy.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
    def main():
        images = filedialog.askopenfilename()
        image = cv2.imread(images)
        image = imutils.resize(image, width=640,height=500)

        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()

        person_count = 0

        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.45:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_count += 1
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")

                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

        cv2.putText(image, f'People: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    main()
def videoDetection():
        # Import the math library for the Euclidean distance calculation
    path = filedialog.askopenfilename()

    protopath = "MobileNetSSD_deploy.prototxt"
    modelpath = "MobileNetSSD_deploy.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

    # Open the video file
    video = cv2.VideoCapture(path)

    # ...

    # Initialize a list to store the coordinates of the people
    people = []

    # ...

    while True:
        # Read the frame
        ret, frame = video.read()

        # Check if the video is over
        if not ret:
            break

        # Resize the frame
        frame = imutils.resize(frame, width=1200)

        # Get the frame dimensions
        (H, W) = frame.shape[:2]

        # Convert the frame to a blob
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        # Pass the blob through the network
        detector.setInput(blob)
        person_detections = detector.forward()

        # Initialize the person count
        person_count = 0

        # Clear the list of people
        people.clear()

        # Loop over the detections
        for i in np.arange(0, person_detections.shape[2]):
            # Extract the confidence and class index
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.6:
                idx = int(person_detections[0, 0, i, 1])

                # If the class is not "person", skip it
                if CLASSES[idx] != "person":
                    continue

                # Extract the bounding box coordinates
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

                # Append the coordinates of the person to the list
                people.append((startX, startY, endX, endY))

                # Increment the person count
                person_count += 1

        # Put the person count on the frame
        cv2.putText(frame, f"Count now: {person_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        

        # Display the frame
        cv2.imshow("Results", frame)

        # Check if the user pressed "q" to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release the video capture and destroy the windows
    video.release()
    cv2.destroyAllWindows()

def countEntre():
    path = filedialog.askopenfilename()
    cnt_up   = 0
    cnt_down = 0
    total_frames = 0
    fps=0



    cap = cv.VideoCapture(path)



    h = 300
    w = 640
    frameArea = h*w
    areaTH = frameArea/250
    print( 'Area Threshold', areaTH)
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    #Lineas de entrada/salida
    line_up = int(2*(h/5))
    line_down   = int(3*(h/5))

    up_limit =   int(1*(h/5))
    down_limit = int(4*(h/5))

    print( "Red line y:",str(line_down))
    print( "Blue line y:", str(line_up))
    line_down_color = (255,0,0)
    line_up_color = (0,0,255)
    pt1 =  [0, line_down];
    pt2 =  [w, line_down];
    pts_L1 = np.array([pt1,pt2], np.int32)
    pts_L1 = pts_L1.reshape((-1,1,2))
    pt3 =  [0, line_up];
    pt4 =  [w, line_up];
    pts_L2 = np.array([pt3,pt4], np.int32)
    pts_L2 = pts_L2.reshape((-1,1,2))

    pt5 =  [0, up_limit];
    pt6 =  [w, up_limit];
    pts_L3 = np.array([pt5,pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1,1,2))
    pt7 =  [0, down_limit];
    pt8 =  [w, down_limit];
    pts_L4 = np.array([pt7,pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1,1,2))

    #Substractor de fondo
    fgbg = cv.createBackgroundSubtractorMOG2(detectShadows = True)

    #Elementos estructurantes para filtros morfoogicos
    kernelOp = np.ones((3,3),np.uint8)
    kernelOp2 = np.ones((5,5),np.uint8)
    kernelCl = np.ones((11,11),np.uint8)

    #Variables
    font = cv.FONT_HERSHEY_SIMPLEX
    persons = []
    max_p_age = 5
    pid = 1

    while(cap.isOpened()):
    ##for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        ret, frame = cap.read()
        total_frames = total_frames + 1
    ##    frame = image.array

        for i in persons:
            i.age_one() #age every person one frame
        
        fgmask = fgbg.apply(frame)
        fgmask2 = fgbg.apply(frame)

        try:
            ret,imBin= cv.threshold(fgmask,200,255,cv.THRESH_BINARY)
            ret,imBin2 = cv.threshold(fgmask2,200,255,cv.THRESH_BINARY)
            mask = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernelOp)
            mask2 = cv.morphologyEx(imBin2, cv.MORPH_OPEN, kernelOp)
            mask =  cv.morphologyEx(mask , cv.MORPH_CLOSE, kernelCl)
            mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernelCl)
        except:
            print('EOF')
            print( 'UP:',cnt_up)
            print ('DOWN:',cnt_down)
            break
        
        
        contours0, hierarchy = cv.findContours(mask2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours0:
            area = cv.contourArea(cnt)
            if area > areaTH:
                #################
                #   TRACKING    #
                #################
                
                
                M = cv.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                x,y,w,h = cv.boundingRect(cnt)

                new = True
                if cy in range(up_limit,down_limit):
                    for i in persons:
                        if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h:
                           
                            new = False
                            i.updateCoords(cx,cy)   
                            if i.going_UP(line_down,line_up) == True:
                                cnt_up += 1;
                                print( "ID:",i.getId(),'crossed going up at',time.strftime("%c"))
                                
                                
                            elif i.going_DOWN(line_down,line_up) == True:
                                cnt_down += 1;
                                print( "ID:",i.getId(),'crossed going down at',time.strftime("%c"))
                                
                                
                            break
                        if i.getState() == '1':
                            if i.getDir() == 'down' and i.getY() > down_limit:
                                i.setDone()
                            elif i.getDir() == 'up' and i.getY() < up_limit:
                                i.setDone()
                        if i.timedOut():
                            index = persons.index(i)
                            persons.pop(index)
                            del i     
                    if new == True:
                        p = Person.MyPerson(pid,cx,cy, max_p_age)
                        persons.append(p)
                        pid += 1     
                
                cv.circle(frame,(cx,cy), 5, (0,0,255), -1)
                img = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)            
                
                
                
        for i in persons:

            cv.putText(frame, str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv.LINE_AA)
        
        
        str_up = 'UP: '+ str(cnt_up)
        str_down = 'DOWN: '+ str(cnt_down)
        cnt_t='total Persons'+str(cnt_down+cnt_up)
        frame = cv.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
        frame = cv.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
        frame = cv.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
        frame = cv.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
        cv.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv.LINE_AA)
        cv.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv.LINE_AA)
        cv.putText(frame, str_down ,(10,270),font,0.5,(255,255,255),2,cv.LINE_AA)
        cv.putText(frame, str_down ,(10,270),font,0.5,(255,0,0),1,cv.LINE_AA)
        cv.putText(frame, cnt_t,(280,40),font,0.5,(0,0,0),2,cv.LINE_AA)
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
                fps = (total_frames / time_diff.seconds)
        fps_text = "FPS: {:.2f}".format(fps)
        cv.putText(frame, fps_text, (5, 10), font, 0.5, (0, 255, 255), 1)
        

        cv.imshow('Frame',frame)
        


        k = cv.waitKey(30) & 0xff
        if k == ord('q'):
            break

        

    cap.release()
    cv.destroyAllWindows()
def socialDistance():
    path = filedialog.askopenfilename()
    protopath = "MobileNetSSD_deploy.prototxt"
    modelpath = "MobileNetSSD_deploy.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)



    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

    tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)


    def non_max_suppression_fast(boxes, overlapThresh):
        try:
            if len(boxes) == 0:
                return []

            if boxes.dtype.kind == "i":
                boxes = boxes.astype("float")

            pick = []

            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            idxs = np.argsort(y2)

            while len(idxs) > 0:
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)

                overlap = (w * h) / area[idxs[:last]]

                idxs = np.delete(idxs, np.concatenate(([last],
                                                    np.where(overlap > overlapThresh)[0])))

            return boxes[pick].astype("int")
        except Exception as e:
            print("Exception occurred in non_max_suppression : {}".format(e))


    def main():
        cap = cv2.VideoCapture(path)

        fps_start_time = datetime.datetime.now()
        fps = 0
        total_frames = 0

        while True:
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=1200)
            total_frames = total_frames + 1

            (H, W) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

            detector.setInput(blob)
            person_detections = detector.forward()
            rects = []
            for i in np.arange(0, person_detections.shape[2]):
                confidence = person_detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(person_detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = person_box.astype("int")
                    rects.append(person_box)

            boundingboxes = np.array(rects)
            boundingboxes = boundingboxes.astype(int)
            rects = non_max_suppression_fast(boundingboxes, 0.3)
            centroid_dict = dict()
            total=0
            objects = tracker.update(rects)
            log.basicConfig(filename='warning.log',level=log.INFO)
            for (objectId, bbox) in objects.items():
                x1, y1, x2, y2 = bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)
                total=total+1


                centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)

                text = "ID: {}".format(objectId)
                cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
                

            red_zone_list = []
            for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
                dx, dy = p1[0] - p2[0], p1[1] - p2[1]
                distance = math.sqrt(dx * dx + dy * dy)
                print(f'Distance between person {id1} and person {id2}: {distance} pixels ',time.strftime("%c"))
                if distance < 140:
                    if id1 not in red_zone_list:
                        red_zone_list.append(id1)
                    if id2 not in red_zone_list:
                        red_zone_list.append(id2)
                
                    
                    log.info("WARNING beetwin person : "+str(id1)+  "   and     "   +str(id2)+    "    at    "+str(time.strftime("%c")))

                    #engine.say(f"Warning in person {id1} and {id2}")
                    #engine.runAndWait()
            

            for id, box in centroid_dict.items():
                if id in red_zone_list:
                    cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
                    
                else:
                    cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)


            fps_end_time = datetime.datetime.now()
            time_diff = fps_end_time - fps_start_time
            if time_diff.seconds == 0:
                fps = 0.0
            else:
                fps = (total_frames / time_diff.seconds)

            fps_text = "FPS: {:.2f}".format(fps)

            cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            cv2.putText(frame, f'Total Persons : {total }', (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0,0), 2)

            cv2.imshow("Application", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


    main()
def LiveDetection():
        

    protopath = "MobileNetSSD_deploy.prototxt"
    modelpath = "MobileNetSSD_deploy.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
    # Only enable it if you are using OpenVino environment
    # detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    # detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

    tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


    def non_max_suppression_fast(boxes, overlapThresh):
        try:
            if len(boxes) == 0:
                return []

            if boxes.dtype.kind == "i":
                boxes = boxes.astype("float")

            pick = []

            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            idxs = np.argsort(y2)

            while len(idxs) > 0:
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)

                overlap = (w * h) / area[idxs[:last]]

                idxs = np.delete(idxs, np.concatenate(([last],
                                                    np.where(overlap > overlapThresh)[0])))

            return boxes[pick].astype("int")
        except Exception as e:
            print("Exception occurred in non_max_suppression : {}".format(e))


    def main():
        cap = cv2.VideoCapture(0)

        fps_start_time = datetime.datetime.now()
        fps = 0
        total_frames = 0

        while True:
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=600)
            total_frames = total_frames + 1

            (H, W) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

            detector.setInput(blob)
            person_detections = detector.forward()
            rects = []
            for i in np.arange(0, person_detections.shape[2]):
                confidence = person_detections[0, 0, i, 2]
                if confidence > 0.7:
                    idx = int(person_detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = person_box.astype("int")
                    rects.append(person_box)

            boundingboxes = np.array(rects)
            boundingboxes = boundingboxes.astype(int)
            rects = non_max_suppression_fast(boundingboxes, 0.3)

            objects = tracker.update(rects)
            for (objectId, bbox) in objects.items():
                x1, y1, x2, y2 = bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 2)
                text = "Num:{}".format(objectId+1)
                cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (100, 100, 100), 1)

            fps_end_time = datetime.datetime.now()
            time_diff = fps_end_time - fps_start_time
            if time_diff.seconds == 0:
                fps = 0.0
            else:
                fps = (total_frames / time_diff.seconds)

            fps_text = "FPS: {:.2f}".format(fps)

            cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

            cv2.imshow("Detectation Des Personnes", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        print(format(objectId+1))
        cv2.destroyAllWindows()


    main()





OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\salman\Desktop\SLIMAN_SALMAN_MQL\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)






def btn_clicked():
    print("Button Clicked")


window = Tk()

window.geometry("1000x600")
window.configure(bg = "#0fcba5")
canvas = Canvas(
    window,
    bg = "#0fcba5",
    height = 600,
    width = 1000,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    500.0, 300.0,
    image=background_img)

img0 = PhotoImage(file = f"img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = picture,
    relief = "flat")

b0.place(
    x = 672, y = 121,
    width = 219,
    height = 46)

img1 = PhotoImage(file = f"img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = videoDetection,
    relief = "flat")

b1.place(
    x = 672, y = 199,
    width = 219,
    height = 46)

img2 = PhotoImage(file = f"img2.png")
b2 = Button(
    image = img2,
    borderwidth = 0,
    highlightthickness = 0,
    command = LiveDetection,
    relief = "flat")

b2.place(
    x = 672, y = 277,
    width = 219,
    height = 46)

img3 = PhotoImage(file = f"img3.png")
b3 = Button(
    image = img3,
    borderwidth = 0,
    highlightthickness = 0,
    command =socialDistance,
    relief = "flat")

b3.place(
    x = 672, y = 355,
    width = 219,
    height = 46)

img4 = PhotoImage(file = f"img4.png")
b4 = Button(
    image = img4,
    borderwidth = 0,
    highlightthickness = 0,
    command =countEntre,
    relief = "flat")

b4.place(
    x = 672, y = 433,
    width = 219,
    height = 46)

window.resizable(True, True)
window.mainloop()

