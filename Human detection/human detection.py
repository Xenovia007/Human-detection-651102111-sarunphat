import cv2

cap = cv2.VideoCapture('e.mp4')
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, 1.7, 1)

    # Resize frame for smoother processing
    resized_frame = cv2.resize(frame, (640, 480))
    resized_gray = cv2.resize(gray, (640, 480))

    # Display the resulting frame
    for (x, y, w, h) in humans:
        # Scale bounding box coordinates to resized frame
        scaled_x = int(x * (640 / frame.shape[1]))
        scaled_y = int(y * (480 / frame.shape[0]))
        scaled_w = int(w * (640 / frame.shape[1]))
        scaled_h = int(h * (480 / frame.shape[0]))

        # Draw rectangle on resized frame
        cv2.rectangle(resized_frame, (scaled_x, scaled_y), (scaled_x + scaled_w, scaled_y + scaled_h), (255, 0, 0), 2)

        # Filter bounding boxes
        if scaled_w < 100 or scaled_h < 100:
            continue

    cv2.imshow('frame', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
