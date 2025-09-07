import cv2
import time
import sys


def main(folder):
    video_capture = cv2.VideoCapture(0)

        
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            print("Exiting...")
            break
        if key  == ord('s'):
            timestr = time.strftime("%Y-%m-%d %H:%M:%S")
            print("Saving image: "+timestr + ".jpeg")
            filename = folder + timestr + ".jpeg"
            cv2.imwrite(filename, frame)
        if key == 13: ## enter key
            timestr = time.strftime("%Y-%m-%d %H:%M:%S")
            print("Saving image: "+timestr + ".jpeg")
            filename = folder + timestr + ".jpeg"
            cv2.imwrite(filename, frame)

    video_capture.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    print("hello")
    folder = sys.argv[1]
    main(folder)
    