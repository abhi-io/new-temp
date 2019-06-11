# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os,gc
from colorama import Fore, Back, Style
start_time = time.time()
temp_pos=[]
# flag1=0
# updatetxt="wait"
camerafeed='cup.webm'
# camerafeed=0


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()


ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.2,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

#------------------------------------------------------------------
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["/home/abc/from-penD/imp/2/yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["/home/abc/from-penD/imp/2/yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["/home/abc/from-penD/imp/2/yolo-coco", "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#------------------------------------------------------------------

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs =cv2.VideoCapture(camerafeed)#########################################

def toDATABASE():
	
	print(Fore.BLUE + "[info] DATABASE comlete")
	writer = None
	(W, H) = (None, None)
	old_items= items
	items_search=[]
	item_pos=[]
	countx=0
	while True:
		
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			print("[camera dead]")
			break
		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]
		#------------------------------------------------------------------
		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()
		# initialize our lists of detected bounding boxes, confidences,
		# and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []
		#------------------------------------------------------------------
		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability)
				# of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > args["confidence"]:
					# scale the bounding box coordinates back relative to
					# the size of the image, keeping in mind that YOLO
					# actually returns the center (x, y)-coordinates of
					# the bounding box followed by the boxes' width and
					# height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					# use the center (x, y)-coordinates to derive the top
					# and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)
		#------------------------------------------------------------------
		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
			args["threshold"])
		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				# draw a bounding box rectangle and label on the frame
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				claName=LABELS[classIDs[i]]
				text = "{}: {:.4f}".format(claName,confidences[i])
				cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				cv2.putText(frame, "[calibration]", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 224, 25), 2)
				# temparray=x,y,w,h
				# transortarray.append(temparray)
				item_pos.append(x)
				items_search.append(claName)
				# count=count+1
				# print(y)
				print(items_search)
				for i in range(len(item_pos)):
					for j in range(i + 1, (len(item_pos))):
						if(item_pos[i] == 0):
							break
						# print(item_pos)
						change_percent = ((item_pos[j])-(item_pos[i]))
						# print(item_pos[i],"--",item_pos[j],"-->>",(change_percent))
						index=0
						if(change_percent> -5 and change_percent<10 ):
							# print("change",item_pos[j])
							item_pos[j]=0
						else:
							index=index+1
						# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",item_pos)
				for i in range(len(item_pos)):
					if(item_pos[i] != 0):
						index = item_pos.index(item_pos[i])
						# print(index)
				for j in range(len(items_search)):
					if(j<=index):
						print("")
					else:
						items_search[j]=0
				countx=countx+1
				print("----------------",countx)
				if (countx == 10):
					print("calibration complete")
					if(len(items_search)==len(items)):
						print("item returned")
						import mysql.connector
						mydb = mysql.connector.connect(host="127.0.0.1",  user="root", passwd="12", database="shopdb")
						mycursor = mydb.cursor()
						sql = "DELETE FROM items WHERE item = %s"
						val = (items_search,)
						mycursor.execute(sql, val)

						mydb.commit()
						print(mycursor.rowcount, "record DELETED.")
						exit()
					else:
						tracking()
					gc.collect()
				cv2.imshow('image',frame)
				if cv2.waitKey(1) == 27:
					break
					vs.release()
					cv2.destroyAllWindows()
	#----------------------------------------------------------------
	# release the file pointers
	print("[INFO] cleaning up...")
	#writer.release()
	vs.release()

def shelfcame():
	
	print(Fore.GREEN + "[info] detection starts")
	writer = None
	(W, H) = (None, None)
	count=0
	items=[]
	item_pos=[]
	transortarray=[]
	temparray=[]
	

	# try to determine the total number of frames in the video file

	#------------------------------------------------------------------
	# loop over frames from the video file stream
	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()

		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			print("[cam dead]")
			break

		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]


		#------------------------------------------------------------------
		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# initialize our lists of detected bounding boxes, confidences,
		# and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		#------------------------------------------------------------------
		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability)
				# of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > args["confidence"]:
					# scale the bounding box coordinates back relative to
					# the size of the image, keeping in mind that YOLO
					# actually returns the center (x, y)-coordinates of
					# the bounding box followed by the boxes' width and
					# height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top
					# and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)
		#------------------------------------------------------------------
		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
			args["threshold"])

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# draw a bounding box rectangle and label on the frame
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				claName=LABELS[classIDs[i]]
				text = "{}: {:.4f}".format(claName,confidences[i])
				cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				cv2.putText(frame, "[calibration]", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 224, 25), 2)
				temparray=x,y,w,h
				transortarray.append(temparray)
				item_pos.append(x)
				items.append(claName)
				# count=count+1
				# print(y)
				print(items)
				for i in range(len(item_pos)):
					for j in range(i + 1, (len(item_pos))):
						if(item_pos[i] == 0):
							break
						# print(item_pos)
						change_percent = ((item_pos[j])-(item_pos[i]))
						# print(item_pos[i],"--",item_pos[j],"-->>",(change_percent))
						index=0
						if(change_percent> -5 and change_percent<10 ):
							# print("change",item_pos[j])
							item_pos[j]=0
						else:
							index=index+1
						# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",item_pos)
				for i in range(len(item_pos)):
					if(item_pos[i] != 0):
						index = item_pos.index(item_pos[i])
						# print(index)
				for j in range(len(items)):
					if(j<=index):
						print("")
					else:
						items[j]=0
				# print(items)

				# # change_percent = ((float(current)-previous)/previous)*100
				# item_pos.append(x)
				# print(item_pos)
				# # temp1=(text.split(': '))
				# # classn=temp1[0]
				# # print(classn)
				# #
				# # items.append(classn)
				# # print(items)
				# #print(centerY)
				count=count+1
				print("----------------",count)
				if (count == 5):
					cv2.destroyAllWindows()
					print("calibration complete")
					return items,item_pos,transortarray,index
					vs.release()
					gc.collect()
					exit()
					

				cv2.imshow('image',frame)

				
				if cv2.waitKey(33) == ord('a'):
					break
					vs.release()
					cv2.destroyAllWindows()

	#----------------------------------------------------------------
	# release the file pointers
	print("[INFO] cleaning up...")
	#writer.release()
	
	

#------------------------------------------------------------------
def tracking():
	# items,item_pos=test()
	flag1=0
	updatetxt="wait"

	items,item_pos,transortarray,index=shelfcame()
	print(Fore.RED + "[info] tracking starts")
	
	global items
	print("transport->>",transortarray)
	print(items,item_pos,"888888",index)
	
	
	trackerTypes = [
	    'BOOSTING',
	    'MIL',
	    'KCF',
	    'TLD',
	    'MEDIANFLOW<<abhi',
	    'GOTURN',
	    'MOSSE<<abhi',
	    'CSRT',
	    ]
	trackerType = 'CSRT'

	def createTrackerByName(trackerType):
	  # Create a tracker based on tracker name
	    if trackerType == trackerTypes[0]:
	        tracker = cv2.TrackerBoosting_create()
	    elif trackerType == trackerTypes[1]:
	        tracker = cv2.TrackerMIL_create()
	    elif trackerType == trackerTypes[2]:
	        tracker = cv2.TrackerKCF_create()
	    elif trackerType == trackerTypes[3]:
	        tracker = cv2.TrackerTLD_create()
	    elif trackerType == trackerTypes[4]:
	        tracker = cv2.TrackerMedianFlow_create()
	    elif trackerType == trackerTypes[5]:
	        tracker = cv2.TrackerGOTURN_create()
	    elif trackerType == trackerTypes[6]:
	        tracker = cv2.TrackerMOSSE_create()
	    elif trackerType == trackerTypes[7]:
	        tracker = cv2.TrackerCSRT_create()
	    else:
	        tracker = None
	        print('Incorrect tracker name')
	    return tracker
	#/////////////////////////////////////////////////////////
	
	# Create a video capture object to read videos
	# cap = cv2.VideoCapture(camerafeed)
	print(Style.RESET_ALL)
	print("//tracking// cam 1")
	# Read first frame
	(g, frame) = vs.read()
	if (g == False):
		print("camera error at [def tacking]")
	bboxes = []
	colors = []
	# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
	# So we will call this function in a loop till we are done selecting all objects

	    #bbox = cv2.selectROI('dawbox', frame)
	  # draw bounding boxes over objects
	  # selectROI's default behaviour is to draw box starting from the center
	  # when fromCenter is set to false, you can draw box starting from top left corner
	
	bbox = transortarray
	for  i in range (3):
	    bboxes.append(bbox[i])
	    print(">----------->",bboxes)
	    print(len(bboxes))
	    # print('Press q to quit selecting boxes and start tracking')
	    # print('Press any other key to select next object')
	    # k = cv2.waitKey(0) & 0xFF
	    # if k == 113:  # q is pressed
	        # break
	print('Selected bounding boxes {}'.format(bboxes))
	# Specify the tracker type
	# Create MultiTracker object
	multiTracker = cv2.MultiTracker_create()
	# Initialize MultiTracker
	for bbox in bboxes:
	    multiTracker.add(createTrackerByName(trackerType), frame, bbox)
	# Process video and track objects
	while vs.isOpened():
	    (_, frame) = vs.read()

	  # get updated location of objects in subsequent frames
	    (_, boxes) = multiTracker.update(frame)
	  # draw tracked objects
	    for (i, newbox) in enumerate(boxes):
	        p1 = (int(newbox[0]), int(newbox[1]))
	        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
	        cv2.rectangle(frame,p1,p2,(12, 1, 165),2,1)
	        print(p1,"--",p2)
	        temp_pos.append(int(newbox[0]))
	        if(flag1>0):
	            for j in range (3):#n
	                diff = (int(newbox[0]) - temp_pos[i])
	                if(diff>5 or diff < -5):#if item misssing
	                    print (diff,">>---------->>",i)
	                    # print(items[i],"missing")
	                    updatetxt= items[i]+" missing"
	                    toDBitem = items[i]
######################################################################
#####################################################################
################# write to db ############################
	                    print("//////////////toDATABASE ---> ")
	                    import mysql.connector
	                    import random
	                    rand_slno = (int(random.uniform(10, 10000)))
	                    mydb = mysql.connector.connect(host="127.0.0.1",  user="root", passwd="12", database="shopdb")
	                    mycursor = mydb.cursor()
	                    sql = "INSERT INTO items (slno,user,item,quantity,price) VALUES (%s, %s,%s, %s,%s)"
	                    val = (rand_slno,"Abhi", toDBitem,"1","50")
	                    mycursor.execute(sql, val)

	                    mydb.commit()
	                    print(mycursor.rowcount, "record inserted.")
	                    toDATABASE()
	                    
	                

	  # show frame+
	    cv2.putText(frame, updatetxt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 224, 25), 2)
	    cv2.imshow('show', frame)
	    print(updatetxt,"!!!!!!!!!!!!!!!")
	    if(flag1==0):
	        t1 = temp_pos
	        flag1=5
	        print("444444444444444444444444444444444444444444444")
	    
	    print("--- %s seconds ---" % (round(time.time() - start_time)))
	  # quit on ESC button
	    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
	        break


	
def test():
	items =['cup', 'cup', 0, 0, 0, 0, 0, 0, 0, 0]
	item_pos = [347, 54, 0, 0, 0, 0, 0, 0, 0, 0]
	return items,item_pos



#------------------------------------------------------------------
def main():
	
	print("main() >> tracking() >> detection() >> toDB()")
	# test()
	# shelfcame()
	tracking()
main()
#------------------------------------------------------------------
#------------------------------------------------------------------
