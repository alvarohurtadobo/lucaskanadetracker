import cv2
import numpy as np

from background import Background
from lucaskanadetracker import LucasKanadeTracker

if __name__ == "__main__":
    current_video = '/home/alvaro/trafficFlow/trialVideos/sar.mp4'
    cap = cv2.VideoCapture(current_video)
    ret, frame = cap.read()
    frame = cv2.resize(frame,(192,108))
    background = Background()
    lucas = LucasKanadeTracker(frame)

    colours = np.random.rand(32, 3)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame,(192,108))
        detections = background.detect(frame)
        my_objects = lucas.update(detections,frame)
        for one_object in my_objects:
            (x0, y0, x1, y1) = np.array(one_object['box']).astype(np.int32)
            object_id = one_object['id']
            frame = cv2.rectangle(  frame,
                                    (x0, y0),
                                    (x1, y1),
                                    colours[object_id % 32, :] * 255,
                                    5)

            frame = cv2.putText(frame,
                                'ID : %d' % (object_id),
                                (x0 - 10, y0 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                colours[object_id % 32, :] * 255, 2)
        ch = cv2.waitKey(1)
        if ch == ord('q'):
            break
