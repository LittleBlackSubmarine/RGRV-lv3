import cv2 as cv


def capture_img(cam_no):
    key = cv.waitKey(1)
    webcam = cv.VideoCapture(cam_no)
    print("\nPress 's' to take an image of object of interest!")
    while True:
        check, frame = webcam.read()
        cv.imshow("Capturing", frame)
        key = cv.waitKey(1)
        if key == ord('s'):
            cv.imwrite('edge_detection.jpg', frame)
            webcam.release()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imwrite("edge_detection_gray.jpg", gray)
            cv.waitKey(1650)
            cv.destroyAllWindows()
            print("Image saved!")
            break
        elif key == ord('q'):
            print("\nTurning off camera.")
            webcam.release()
            print("Camera off.")
            cv.destroyAllWindows()
            break














































