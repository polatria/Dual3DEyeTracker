import uvc
import cv2
import numpy as np
import sys
import time
import threading
from msvcrt import getch
import EyeDetector

EYES = ['Right', 'Left']  # Window names
TERMINATE = 27  # ESC key

# Setup variables
cap = [0] * 2  # Camera image sources
img = [0] * 2  # Image buffers
key = 0  # global key status
lock = threading.Lock()  # lock object


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

print(cap[0].avaible_modes)

# Eye detector init
trk = EyeDetector.alloc()

# for FPS calculation
tm = cv2.TickMeter()
tm.start()
count = 0
max_count = 100
fps = 0

# Main loop
threading.Thread(target=key_check).start()
is_run = True
while is_run:
    if key == TERMINATE:
        is_run = False
        break

    # Fetch images
    for i in range(len(EYES)):
        img[i] = cap[i].get_frame_robust().img  # ndarray (cvMat)

    # Process each camera images
    for i in range(len(EYES)):
        EyeDetector.detect(trk, img[i], i, key)
        # cv2.imshow(EYES[i], img[i])

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

cv2.destroyAllWindows()
cap = None
