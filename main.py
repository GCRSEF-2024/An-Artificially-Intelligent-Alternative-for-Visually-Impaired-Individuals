import cv2
#from cv2 import cv2
import numpy as np
from PIL import Image
import os
import time
import board
from adafruit_motorkit import MotorKit
from src.modules import DTT
hull=None
#motorkit object and references
kit = MotorKit(i2c=board.I2C())
MotorL=kit.motor1
MotorR=kit.motor2


def ThrottleAllMotors(throttle: float):
    MotorL.throttle = throttle
    MotorR.throttle = throttle

def EStop():
    ThrottleAllMotors(0)
    print("MCS: ***Emergency Stop*** - ", time.strftime("%H:%M:%S"))

# For visual confirmation of motor functionality
def MotorTest():
    MotorL.throttle = 1
    time.sleep(0.1)
    MotorL.throttle = 0

    MotorR.throttle = 1
    time.sleep(0.1)
    MotorR.throttle = 0

    ThrottleAllMotors(1)
    time.sleep(0.1)
    ThrottleAllMotors(-1)
    time.sleep(0.1)
    ThrottleAllMotors(0)

    print("MCS: Movement Control System nominal.")
# read image
cap = cv2.VideoCapture('ttt2.mp4')
#cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
count = 0

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Adjust the width as needed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Adjust the height as needed
cv2.waitKey(500)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
while cap.isOpened():

	ret, img = cap.read()
	if not ret:
		break
	cv2.imshow('Frame', img)
	# threshold on white/gray sidewalk stripes
	lower = (130,150,150)
	upper = (170,190,190)
	thresh = cv2.inRange(img, lower, upper)

	# apply morphology close to fill interior regions in mask
	kernel = np.ones((3,3), np.uint8)
	morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	kernel = np.ones((5,5), np.uint8)
	morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

	# get contours
	cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# bounding_boxes = [cv2.boundingRect(contour) for contour in cntrs]
	cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

	# filter on area
	contours = img.copy()
	#good_contours = []
	#for c in cntrs:
	    #area = cv2.contourArea(c)
	    #if area > 200:
	        #cv2.drawContours(contours, [c], -1, (0,0,255), 1)
	        #good_contours.append(c)
	good_contours = []
	for c in cntrs:
	    area = cv2.contourArea(c)
	    if 30 < area < 400000:  # Adjust the area range based on your specific case
	        rect = cv2.minAreaRect(c)
	        aspect_ratio = rect[1][0] / rect[1][1]
	        width = min(rect[1])  # Get the minimum width of the bounding box

	        # Adjust these values based on your specific case
	        if 0.2 < aspect_ratio < 10.0 and width > 10 and abs(rect[2]) < 100 and aspect_ratio > 2:
	            cv2.drawContours(contours, [c], -1, (0, 0, 255), 1)
	            good_contours.append(c)

	# combine good contours
	if good_contours:
		contours_combined = np.vstack(good_contours)
		bounding_boxes = [cv2.boundingRect(contour) for contour in contours_combined]
		print(bounding_boxes)
		hull  = cv2.convexHull(contours_combined)
	else:
		print("No valid contours found.")
	# get convex hull
	#result = img.copy()
	#hull = cv2.convexHull(contours_combined)
	#cv2.polylines(result, [hull], True, (0,0,255), 2)
	# write result to disk
	#cv2.imwrite("walkway_thresh.jpg", thresh)
	#cv2.imwrite("walkway_morph.jpg", morph)
	#cv2.imwrite("walkway_contours.jpg", contours)
	#cv2.imwrite("walkway_result.jpg", result)

	# display it
	#cv2.imshow("THRESH", thresh)
	#cv2.imshow("MORPH", morph)
	#cv2.imshow("CONTOURS", contours)
	#cv2.imshow("RESULT", result)
	#cv2.waitKey(0)

	# get convex hull
	#result = img.copy()
	#hull = cv2.convexHull(contours_combined)

	# Approximate a polygon around the convex hull
	#epsilon = 0.02 * cv2.arcLength(hull, True)
	#approx = cv2.approxPolyDP(hull, epsilon, True)

	# Draw the approximate polygon on the result image
	#cv2.polylines(result, [approx], True, (0, 0, 255), 2)

	# Get the vertices of the approximate polygon
	#approx_vertices = approx.reshape(-1, 2)

	# Take only the first 4 vertices
	#quadrilateral_vertices = approx_vertices[:4]
	# get convex hull
	result = img.copy()
	#hull = cv2.convexHull(contours_combined)
	cv2.polylines(result, [hull], True, (0, 0, 255), 2)

	# Approximate a polygon around the convex hull
	epsilon = 0.02 * cv2.arcLength(hull, True)
	approx = cv2.approxPolyDP(hull, epsilon, True)

	# Draw the approximate polygon on the result image
	#cv2.polylines(result, [approx], True, (0, 0, 255), 2)

	# Get the vertices of the approximate polygon
	approx_vertices = approx.reshape(-1, 2)

	# Take only the first 4 vertices
	quadrilateral_vertices = approx_vertices[:4]

	# Filter out unwanted points using convexity defects
	#hull = cv2.convexHull(quadrilateral_vertices, returnPoints=False)
	#defects = cv2.convexityDefects(quadrilateral_vertices, hull)

	# Filter out small defects
	#filtered_defects = [defect[0] for defect in defects if defect[0, 3] > 20]

	# Use the convex hull points for the bounding rectangle
	#bounding_rect_points = cv2.convexHull(quadrilateral_vertices, returnPoints=True)
	#bounding_rect = np.int0(bounding_rect_points)

	# Draw the bounding rectangle on the result image
	#cv2.drawContours(result, [bounding_rect], 0, (0, 0, 255), 2)
	# Filter out unwanted points using convexity defects
	hull = cv2.convexHull(quadrilateral_vertices, returnPoints=False)
	defects = cv2.convexityDefects(quadrilateral_vertices, hull)

	if defects is not None:
	    # Filter out small defects
	    filtered_defects = [defect[0] for defect in defects if defect[0, 3] > 20]

	    # Use the convex hull points for the bounding rectangle
	    bounding_rect_points = cv2.convexHull(quadrilateral_vertices, returnPoints=True)
	    bounding_rect = np.int0(bounding_rect_points)

	    # Draw the bounding rectangle on the result image
	    cv2.drawContours(result, [bounding_rect], 0, (0, 0, 255), 2)
	else:
	    print("No defects found. Unable to simplify convex hull.")
	# Print the vertices in unit coordinates
	for vertex in quadrilateral_vertices:
	    print("Vertex:", vertex)

	# Display and save the updated result
	cv2.imshow("UPDATED RESULT", result)
	cv2.imwrite("walkway_updated_result.jpg", result)
	#cv2.waitKey(0)

	# Get the vertices of the approximate polygon
	approx_vertices = approx.reshape(-1, 2)

	# Take only the first 4 vertices
	quadrilateral_vertices = approx_vertices[:4]

	# Assuming quadrilateral_vertices are ordered as [top-left, top-right, bottom-right, bottom-left]
	# Calculate the slope of the long sides
	denominator = quadrilateral_vertices[1][0] - quadrilateral_vertices[0][0]
	if denominator != 0:
    		slope_top = (quadrilateral_vertices[1][1] - quadrilateral_vertices[0][1]) / denominator
	else:
    		slope_top = float('inf')  # Handle division by zero by setting slope to infinity
	#slope_top = (quadrilateral_vertices[1][1] - quadrilateral_vertices[0][1]) / (quadrilateral_vertices[1][0] - quadrilateral_vertices[0][0])
	#slope_bottom = (quadrilateral_vertices[2][1] - quadrilateral_vertices[3][1]) / (quadrilateral_vertices[2][0] - quadrilateral_vertices[3][0])
	denominator = quadrilateral_vertices[2][0] - quadrilateral_vertices[3][0]
	if denominator != 0:
    		slope_bottom = (quadrilateral_vertices[2][1] - quadrilateral_vertices[3][1]) / denominator
	else:
    		slope_bottom = float('inf')  # Handle division by zero by setting slope to infinity
	# Calculate the intersection point of the extended lines
	#intersection_x = (slope_top * quadrilateral_vertices[0][0] - quadrilateral_vertices[0][1] - slope_bottom * quadrilateral_vertices[3][0] + quadrilateral_vertices[3][1]) / (slope_top - slope_bottom)
	#intersection_y = slope_top * (intersection_x - quadrilateral_vertices[0][0]) + quadrilateral_vertices[0][1]
	if slope_top != slope_bottom:
    		intersection_x = (slope_top * quadrilateral_vertices[0][0] - quadrilateral_vertices[0][1] - slope_bottom * quadrilateral_vertices[3][0] + quadrilateral_vertices[3][1]) / (slope_top - slope_bottom)
    		intersection_y = slope_top * (intersection_x - quadrilateral_vertices[0][0]) + quadrilateral_vertices[0][1]
	else:
    		intersection_x, intersection_y = float('nan'), float('nan')  # Handle parallel lines
	# Set the middle of the picture as x=0
	intersection_x -= result.shape[1] // 2
	# Mark the x=0 point at the bottom of the picture
	zero_point = (result.shape[1] // 2, result.shape[0])
	cv2.circle(result, zero_point, 5, (0, 255, 0), -1)

	# Print the intersection point
	print("Intersection Point:", (intersection_x, intersection_y))

	# Draw a circle at the intersection point on the result image
	#cv2.circle(result, (int(intersection_x + result.shape[1] // 2), int(intersection_y)), 5, (255, 0, 0), -1)
	if not np.isnan(intersection_x) and not np.isnan(intersection_y):
	    # Convert to integers only if the values are valid
	    int_intersection_x = int(intersection_x + result.shape[1] // 2)
	    int_intersection_y = int(intersection_y)

	    cv2.circle(result, (int_intersection_x, int_intersection_y), 5, (255, 0, 0), -1)
	else:
	    print("Intersection point is not valid.")


	# Determine the movement direction
	if abs(intersection_x) < 500:
		print("Move straight")
		MotorTest()
		time.sleep(0.5)
	elif intersection_x < 500:
		print("Move left")
		MotorTest()
		time.sleep(0.5)
	else:
		print("Move right")
		MotorTest()
		time.sleep(0.5)
	# Display and save the updated result
	cv2.imshow("UPDATED RESULT", result)
	cv2.imwrite("walkway_updated_result.jpg", result)
	#cv2.waitKey(0)
	if cv2.waitKey(25) & 0xFF == ord('q'):
	        break
cap.release()
cv2.destroyAllWindows()
