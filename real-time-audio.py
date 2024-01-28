import torch
import numpy as np
import cv2
import os
import subprocess
from gtts import gTTS 
from pydub import AudioSegment

# loads yolov5 model using torch hub 
# automatically downloads latest version if doesn't exist in root directory
model = torch.hub.load("ultralytics/yolov5", "yolov5l", pretrained=True)

# gets labels from model
LABELS = model.names

# generates random colors for each label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# starts video capture
vs = cv2.VideoCapture(0)
first = True
detected = []

frame_count = 0

while True:
    print(frame_count)
    frame_count += 1

    # reads frame from video capture
    (grabbed, frame) = vs.read()
    frame = cv2.flip(frame, 1)
    
    # breaks loop if no frame is grabbed
    if not grabbed:
        break

    # processes frame every other frame
    if frame_count % 2 == 0:
        (H, W) = frame.shape[:2]

        # gets results from model
        results = model(frame)

        boxes = []
        confidences = []
        classIDs = []
        centers = []

        # loops through each detection
        for det in results.xyxy[0]:
            classID = int(det[5])
            confidence = float(det[4])

            # if confidence is greater than 0.5, add detection to list
            if confidence > 0.5:
                (x, y, w, h) = map(int, det[:4])
                centerX = (2 * x + w) / 2
                centerY = (2 * y + h) / 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIDs.append(classID)
                centers.append((centerX, centerY))

        # applies non-maxima suppression to remove overlapping boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        texts = []

        # loops through each detection
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])


                # put text on frame 
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if frame_count % 15 == 0:
                    label = LABELS[classIDs[i]]
                    if label not in detected:
                        centerX, centerY = centers[i][0], centers[i][1]

                        if centerX <= W/3:
                            W_pos = "left "
                        elif centerX <= (W/3 * 2):
                            W_pos = "center "
                        else:
                            W_pos = "right "

                        if centerY <= H/3:
                            H_pos = "top "
                        elif centerY <= (H/3 * 2):
                            H_pos = "mid "
                        else:
                            H_pos = "bottom "
                        texts.append(H_pos + W_pos + label)
                        detected.append(label)
        
        #  if texts is not empty, convert to audio and play
        if texts:
            text = ", ".join(texts)
            tts = gTTS(text=text, lang="en")
            tts.save('voice.mp3')
            tts = AudioSegment.from_mp3("voice.mp3")
            subprocess.call(["ffplay", "-nodisp", "-autoexit", "voice.mp3"])

        # show frame
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

#  release video capture and destroy all windows
vs.release()
cv2.destroyAllWindows()
os.remove("voice.mp3")
