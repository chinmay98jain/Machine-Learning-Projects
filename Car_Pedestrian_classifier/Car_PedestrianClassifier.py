import cv2
car_casc='cars.xml'
ped_casc='pedestrian.xml'
car_cascade = cv2.CascadeClassifier(car_casc)
ped_cascade = cv2.CascadeClassifier(ped_casc)

def detect_car_peds(frame):
    cars=car_cascade.detectMultiScale(frame,1.15,4)
    peds = ped_cascade.detectMultiScale(frame, 1.15, 4)
    for(x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),color=(255,0,0),thickness=2)
    for (x, y, w, h) in peds:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    return frame



def working():
    video=cv2.VideoCapture('car_ped.mp4')
    while video.isOpened():
        ret,frame=video.read()
        controlKey=cv2.waitKey(1)
        if ret:
            cars_peds_frame=detect_car_peds(frame)
            cv2.imshow('frame',cars_peds_frame)
        else:
            break
        if controlKey == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    working()
