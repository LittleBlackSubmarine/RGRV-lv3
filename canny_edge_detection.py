import cv2 as cv
import numpy as np
import math as mat


import camera_calibration as cc
import capture_img as cap



def onMouseImageClick(event, x, y, flags, param):
    global click_counter
    if event == cv.EVENT_LBUTTONDBLCLK:
       print("x: "+str(x) + " , y:"+str(y))
       image_points[click_counter][0] = x
       image_points[click_counter][1] = y
       click_counter += 1


click_counter = 0

image_points = np.zeros((4, 2), float)
object_points = np.zeros((4, 3), float)


cam_no = int(input("Choose camera [0 for primary, 1 for secondary]: "))
mtx, dist = cc.camera_calibration(cam_no)  # Camera calibration
cap.capture_img(cam_no)   # Capturing image from webcam

original = cv.imread('edge_detection.jpg')
gray = cv.imread('edge_detection_gray.jpg')

window = 'Grayscale'
cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
cv.setMouseCallback(window, onMouseImageClick, image_points)

print("\nDoubleclick 4 points on image. Selected points will be corners of a new image [click points in proper order!]\n")

while True:
    cv.imshow('Grayscale', gray)
    cv.waitKey(10)
    if click_counter == 4:
        break

#mm ROI
object_points[0][0] = 0.0
object_points[0][1] = 0.0
object_points[0][2] = 0.0

object_points[1][0] = 297.0
object_points[1][1] = 0.0
object_points[1][2] = 0.0

object_points[2][0] = 0.0
object_points[2][1] = 210.0
object_points[2][2] = 0.0

object_points[3][0] = 297.0
object_points[3][1] = 210.0
object_points[3][2] = 0.0

crop_width = int(image_points[3][0] - image_points[0][0])
crop_height = int(image_points[3][1] - image_points[0][1])
u0 = int(image_points[0][0])
v0 = int(image_points[0][1])


ed_cropped = original[v0:v0+crop_height, u0:u0+crop_width]
ed_gray_cropped = cv.cvtColor(ed_cropped, cv.COLOR_BGR2GRAY)


dst = cv.Canny(ed_gray_cropped, 75, 150, 3)

color_dst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
cv.imshow('Cropped image', ed_gray_cropped)
cv.imshow('Canny_edge_detector', color_dst)
cv.waitKey(1650)


lines = np.array(cv.HoughLines(dst, 1, np.pi/180, 75, 0, 0))
global rho, theta

for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(np.round(x0 + 1000 * (-b)))
    y1 = int(np.round(y0 + 1000 * a))
    x2 = int(np.round(x0 - 1000 * (-b)))
    y2 = int(np.round(y0 - 1000 * a))

    cv.line(ed_cropped, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imwrite('houghlines.jpg', ed_cropped)
    cv.imshow('Most dominant detected edge [Hough method]', ed_cropped)
    cv.waitKey(10000)


rho = rho + u0*mat.cos(theta) + v0*mat.sin(theta)

ret, rvec, tvec = cv.solvePnP(object_points, image_points, mtx, dist)

R, jacobian = cv.Rodrigues(rvec)


A = (cv.gemm(mtx, R, 1, 0, 0))
bb = (cv.gemm(mtx, tvec, 1, 0, 0))


_x = A[0][0] * mat.cos(theta) + A[1][0] * mat.sin(theta) - rho * A[2][0]
_y = A[0][1] * mat.cos(theta) + A[1][1] * mat.sin(theta) - rho * A[2][1]
_rho = bb[2] * rho - bb[0] * mat.cos(theta) - bb[1] * mat.sin(theta)


thetaN = cv.fastAtan2(_y, _x)
rhoN = _rho / mat.sqrt(pow(_x, 2) + pow(_y, 2))

print("\n Theta angle: " + str(thetaN) + " degrees")
print("\n Rho distance: " + str(rhoN) + "mm")












