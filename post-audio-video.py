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

# takes in video
vs = cv2.VideoCapture("in.mp4")
first = True
detected = []


width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vs.get(cv2.CAP_PROP_FPS)

# creates a writer to save video   
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter("video.avi", fourcc, 30, (width, height))
frame_count = 0

while True:
    print(frame_count)
    frame_count += 1

    (grabbed, frame) = vs.read()
    frame = cv2.flip(frame, 1)
    
    if not grabbed:
        break

    if frame_count % 2 == 0:
        (H, W) = frame.shape[:2]

        results = model(frame)

        boxes = []
        confidences = []
        classIDs = []
        centers = []

        for det in results.xyxy[0]:
            classID = int(det[5])
            confidence = float(det[4])

            if confidence > 0.5:
                (x, y, w, h) = map(int, det[:4])
                centerX = (2 * x + w) / 2
                centerY = (2 * y + h) / 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIDs.append(classID)
                centers.append((centerX, centerY))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        texts = []

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])

                # draws bounding box and label on frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if frame_count % 30 == 0:
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

        # crates audio file
        if frame_count % 30 == 0:
            if len(texts) > 0:
                silence = AudioSegment.silent(duration=0.01*1000)
                description = ', '.join(texts)
                tts = gTTS(description, lang='en')
                tts.save('tts.mp3')
                tts = AudioSegment.from_mp3("tts.mp3")
                if first:
                    audio = tts
                else:
                    audio = AudioSegment.from_mp3("audio.mp3")
                    audio = audio + silence + tts
            else:
                silence = AudioSegment.silent(duration=1*1000)
                if first:
                    audio = silence
                else:
                    audio = AudioSegment.from_mp3("audio.mp3")
                    audio = audio + silence

            audio.export("audio.mp3", format="mp3")
            first = False

    # writes frame to video 
    writer.write(frame)

# releases writer and video
writer.release()
vs.release()
cv2.destroyAllWindows()
os.remove("tts.mp3")

# combines audio and video
cmd = 'ffmpeg -i video.avi -i audio.mp3 -c:v libx264 -c:a aac -strict experimental -b:a 128k -shortest output.mp4'
subprocess.call(cmd, shell=True)