import cv2
import time
import os
import Hand as hd

pTime= 0
cap= cv2.VideoCapture(0)

FolderPath= 'Fingers'
lst= os.listdir(FolderPath)
lst_2= []

for i in lst:
    img= cv2.imread(f'{FolderPath}/{i}')
    lst_2.append(img)
    
dectector= hd.handDetector(detectionCon=1)

fingersId= [4, 8, 12, 16, 20]
while True:
    ret,frame= cap.read()
    frame= dectector.findHands(frame)
    lmList= dectector.findPosition(frame, draw=False)
    # print(lmList)
    
    
    if len(lmList) != 0 :
        fingers= []
        
        # count thumbFinger
        if lmList[fingersId[0]][1] < lmList[fingersId[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # count otherFingers
        for id in range(1, 5):
            if lmList[fingersId[id]][2] < lmList[fingersId[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        countFingers= fingers.count(1)
    
        h, w, c= lst_2[countFingers-1].shape
        frame[0:h, 0:w]= lst_2[countFingers-1]

        cv2.rectangle(frame, (0, 200), (150, 400), (0, 255, 0), 2)
        cv2.putText(frame, str(countFingers), (30, 350), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 3)
    
    # calculator Fps
    cTime= time.time()
    fps= 1/(cTime-pTime)
    pTime= cTime
    
    # Show Fps
    cv2.putText(frame, f'FPS: {int(fps)}', (150, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1)== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()