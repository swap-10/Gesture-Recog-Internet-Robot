import os
import cv2

camera = cv2.VideoCapture(0)
while True:
    cv2.imshow('image', camera.read()[1])
    cv2.waitKey(1)
    a = input()
    if a == 'q':
        break
cv2.destroyWindow('image')
paths = {"Forward": '.\dataset\\test\Forward\\', "Backward": '.\dataset\\test\Backward\\', "Left": '.\dataset\\test\Left\\', "Right": '.\dataset\\test\Right\\', "Stop": '.\dataset\\test\Stop\\'}

for folder in paths:
    path = paths[folder]
    count = 0
    delay = 0
    print("Press 's' to start dataset creation for" + folder)
    start = input()
    if start != 's':
        continue
    while count < 100:
        status, frame = camera.read()
        if not status:
            print("Frame capture failed.")
            continue
        cv2.imshow("Capturing", frame)
        cv2.waitKey(1)
        if delay >200:
            print("Captured frame", count+1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (150, 150))
            cv2.imwrite(path + folder + "%04d" % (count+1) + ".jpg", frame)
            count = count + 1
        elif delay <=200:
            delay = delay + 1
            print("Get ready", delay+1)

camera.release()
cv2.destroyAllWindows()