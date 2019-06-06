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
    if devices[i]['name'] == 'Pupil Cam1 ID1':
        cap[1] = uvc.Capture(devices[i]['uid'])

# Eye detector init
trk = EyeDetector.alloc()

# Main loop
threading.Thread(target=key_check).start()
is_run = True
while is_run:
    if key == TERMINATE:
        is_run = False
        break

    # Fetch images
    for i in range(len(EYES)):
        img[i] = np.array(cap[i].get_frame_robust().img)

    # Process each camera images
    for i in range(len(EYES)):
        EyeDetector.detect(trk, img[i], i, key)

    cv2.waitKey(1)  # 1 msec

    # Compute FPS

cv2.destroyAllWindows()
