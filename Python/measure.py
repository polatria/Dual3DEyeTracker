import uvc
import cv2
import numpy as np
import sys
import time
import threading
from msvcrt import getch
import EyeDetector
import os
import csv

EYES = ['Right', 'Left']  # Window names
TERMINATE = 27  # ESC key
TRIALTIME = 10  # 試行時間, unit: sec

# Setup variables
cap = [0] * len(EYES)  # Camera image sources
img = [0] * len(EYES)  # Image buffers
key = 0  # global key status
lock = threading.Lock()  # lock object
saveData = [[], []]


def key_check():
    global key
    global lock
    while True:
        # Fetch key input
        lock.acquire()  # lock object
        key = ord(getch())
        lock.release()  # release object
        print(f"key input:{key}")
        if key == TERMINATE:
            print("Closing key_check...")
            time.sleep(1)
            break


# Camera open
devices = uvc.Device_List()
if not devices:
    print("EyeTracker not found")
    sys.exit(1)
for i in range(len(devices)):
    if devices[i]['name'] == 'Pupil Cam1 ID0':
        cap[0] = uvc.Capture(devices[i]['uid'])
        cap[0].frame_mode = (640, 480, 60)
    if devices[i]['name'] == 'Pupil Cam1 ID1':
        cap[1] = uvc.Capture(devices[i]['uid'])
        cap[1].frame_mode = (640, 480, 60)

# Eye detector init
trk = EyeDetector.alloc(len(EYES))

# for FPS calculation
tm = cv2.TickMeter()
tm.start()
count = 0
max_count = 100
fps = 0

input("Press Return key to start measurement...\n")
# Main loop
threading.Thread(target=key_check).start()
is_run = True
ellipse = [[], []]
eyeball = [[], []]
start_time = time.time()
while is_run:
    if key == TERMINATE:
        is_run = False
        break

    # Fetch images
    for i in range(len(EYES)):
        img[i] = cap[i].get_frame_robust().img  # ndarray (cvMat)

    # Process each camera images
    for i in range(len(EYES)):
        pts = EyeDetector.detect(trk, img[i], i, key)
        ellipse[i].append(np.array(pts)[0])
        eyeball[i].append(np.array(pts)[1])

    cv2.waitKey(1)  # Required to display camera image

    # Compute FPS
    if count == max_count:
        tm.stop()
        fps = max_count / tm.getTimeSec()
        print(f"{fps:.1f} fps")
        tm.reset()
        tm.start()
        count = 0
    count += 1

    if time.time() - start_time > TRIALTIME:
        saveData[0].append(ellipse)
        saveData[1].append(eyeball)
        break

cv2.destroyAllWindows()
cap = None

eyeside = ['./Right', './Left']
eyepos = ['/elps', '/eybl']
for i in range(2):
    os.makedirs(eyeside[i], exist_ok=True)
    for j in range(2):
        os.makedirs(eyeside[i] + eyepos[j], exist_ok=True)

dataofs = 0
while True:
    path = eyeside[0] + eyepos[0] + '/' + str(dataofs) + '.csv'
    if os.path.exists(path):
        dataofs += 1
    else:
        break

for n in range(len(saveData[0])):
    for i in range(len(eyeside)):
        for j in range(len(eyepos)):
            path = eyeside[i] + eyepos[j] + f'/{n + dataofs}.csv'
            with open(path, mode='a') as f:
                w = csv.writer(f, lineterminator='\n')
                for k in range(len(saveData[j][n][i])):
                    w.writerow(saveData[j][n][i][k])
