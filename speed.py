import cv2
import time
import math

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('test.mp4')

WIDTH = 1280
HEIGHT = 720

def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed

def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    # Write output to video file
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))

    # List of available trackers
    tracker_types = ['KCF', 'MIL', 'CSRT']
    tracker_type = tracker_types[0]  # Choose your preferred tracker

    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print(f"Tracker type {tracker_type} not available in this OpenCV version.")

    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        frameCounter += 1

        carIDtoDelete = []

        for carID in carTracker.keys():
            ok, bbox = carTracker[carID].update(image)
            if not ok:
                carIDtoDelete.append(carID)
            else:
                # Update tracked position using the bounding box
                x, y, w, h = [int(v) for v in bbox]
                carLocation2[carID] = [x, y, w, h]

        for carID in carIDtoDelete:
            print(f'Removing carID {carID} from list of trackers.')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    if carID in carLocation2:
                        t_x, t_y, t_w, t_h = carLocation2[carID]
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                            matchCarID = carID

                if matchCarID is None:
                    print(f'Creating new tracker {currentCarID}')

                    bbox = (x, y, w, h)
                    if tracker:
                        new_tracker = cv2.TrackerKCF_create()  # Initialize the chosen tracker
                        new_tracker.init(image, bbox)
                        carTracker[currentCarID] = new_tracker
                        carLocation1[currentCarID] = [x, y, w, h]

                        currentCarID += 1

        for carID in carTracker.keys():
            if carID in carLocation2:
                ok, bbox = carTracker[carID].update(image)
                if ok:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(resultImage, (x, y), (x + w, y + h), rectangleColor, 4)

                    carLocation2[carID] = [x, y, w, h]

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0 / (end_time - start_time)

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                if i in carLocation1 and i in carLocation2:
                    [x1, y1, w1, h1] = carLocation1[i]
                    [x2, y2, w2, h2] = carLocation2[i]

                    carLocation1[i] = [x2, y2, w2, h2]

                    if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                        if (speed[i] is None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                            speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                        if speed[i] is not None and y1 >= 180:
                            cv2.putText(resultImage, f"{int(speed[i])} km/hr", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow('result', resultImage)
        out.write(resultImage)

        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    trackMultipleObjects()
