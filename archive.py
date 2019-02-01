import numpy as np
import cv2
img = cv2.imread('g.jpg',1)
img = cv2.line(img,(200,0), (200,500), (255,255,255), 10)

x='abhqi'
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,x,(100,100), font, 1,(255,255,255),2,cv2.LINE_AA)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,x,(200,100), font, 1,(255,255,255),2,cv2.LINE_AA)



cv2.imshow('image',img)
k = (cv2.waitKey(0) & 0xFF == ord('q'))
if k == 27:
# wait for ESC key to exit
	cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
	cv2.imwrite('messigray.png',img)
	cv2.destroyAllWindows()

------ screen capture -----------
import numpy as np 
import cv2 
import pyscreenshot as ImageGrab

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1366, 768))

while True:
	img = ImageGrab.grab()
	img_np = np.array(img)
	frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
#	cv2.imshow("Screen", frame)

	out.write(frame)

	if cv2.waitKey(1) == 27:
		break

out.release()
cv2.destroyAllWindows() 

------ screen capture with mss -----------

import time
import cv2
import mss
import numpy

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1366, 768))

with mss.mss() as sct:
	# Part of the screen to capture
	monitor = {"top": 00, "left": 0, "width": 800, "height": 640}

	while "Screen capturing":
		last_time = time.time()

		# Get raw pixels from the screen, save it to a Numpy array
		img = numpy.array(sct.grab(monitor))

		# Display the picture
		cv2.imshow("OpenCV/Numpy normal", img)

		# Display the picture in grayscale
		# cv2.imshow('OpenCV/Numpy grayscale',
		#            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

		print("fps: {}".format(1 / (time.time() - last_time)))

		# Press "q" to quit
		if cv2.waitKey(25) & 0xFF == ord("q"):
			cv2.destroyAllWindows()
			break



-----------------------

cv2.imshow('image',img)
if cv2.waitKey(0) & 0xFF == ord('q'):
	# When everything done, release the capture

	cap.release()
	cv2.destroyAllWindows()

------- background sub --------
import numpy as np
import cv2
cap = cv2.VideoCapture('vt.mp4')
while(cap.isOpened()):
	subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)

	while True:
		_, frame = cap.read()
		mask = subtractor.apply(frame)

		imS = cv2.resize(frame, (96, 54))
		imS1 = cv2.resize(mask, (96, 54))

		cv2.imshow("Frame", imS)
		cv2.imshow("mask", imS1)

		key = cv2.waitKey(30)
		if key == 27:
			break

cap.release()
cv2.destroyAllWindows()


-------- eye classifier -----
import numpy as np
import cv2

while True:
	
	face_cascade = cv2.CascadeClassifier('parojos.xml')
	img = cv2.imread('face.png')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		

	cv2.imshow('image',img)

	if cv2.waitKey(1) == 27:
		break

out.release()
cv2.destroyAllWindows()

/////////////// working //////////////////
import numpy as np
import cv2
import time

face_cascade = cv2.CascadeClassifier('parojos.xml') #eye
cap = cv2.VideoCapture('sh.mp4')
print("1")
time.sleep(1)
while(cap.isOpened()):

	print("2")
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print("3")
	for (x,y,w,h) in faces:
		print("qqqqqqqqqqqqqq")
		img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		print("qqqqqqqqqqqqqq")

		cv2.imshow('image',img)
		cv2.imshow('image',frame)

		if cv2.waitKey(1) == 27:
			break
print("no video")

out.release()
cv2.destroyAllWindows()

//////////////// opencv- o/p frames of a video////////////////

import cv2     # for capturing videos
import math   # for mathematical operations


count = 0
videoFile = "traffic.mp4"
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
frameRate = cap.get(5) #frame rate
x=1
while(cap.isOpened()):
	frameId = cap.get(1) #current frame number
	ret, frame = cap.read()
	if (ret != True):
		break
	if (frameId % math.floor(frameRate) == 0):
		filename ="frame%d.jpg" % count;count+=1
		cv2.imwrite(filename, frame)
		print ("Done!",count)
cap.release()
print ("Done!")

print ("okk")


////////////////// complte cv cascader //////////////////////////
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('palm.xml')
cap = cv2.VideoCapture(0)
print("1")
while(cap.isOpened()):

	print("2")
	_, frame = cap.read()
	
	img = frame
	#img = cv2.imread('palm-2.jpg')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.1, 1)
	print("3")
	for (x,y,w,h) in faces:
		print(x,y,w,h)
			
		img1 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),30)
		#img2 = cv2.rectangle(img1,(w,h),(w+2,h+2),(0,255,0),30)
		cv2.imshow('image',img1)

	if cv2.waitKey(1) == 27:
		break

out.release()
cv2.destroyAllWindows()
