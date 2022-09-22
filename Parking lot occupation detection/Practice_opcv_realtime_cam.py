import cv2
import time
video = cv2.VideoCapture(0)
fps = video.get(cv2.CAP_PROP_FPS)
print('fsp: ',fps)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Video size (width, height) :',size)
while True:
    ret, frame = video.read()
    cv2.imshow("A video", frame)
    c = cv2.waitKey(10) # video refresh rate # 1 millisecond, 1000 millisecond = 1 second
    if c == 27: # press  'esc' to stop
        break
time.sleep(1) # sleep 1 sec
video.release()
cv2.destroyAllWindows() # close the window
