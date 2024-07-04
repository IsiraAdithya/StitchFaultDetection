import cv2
import numpy as np

video_path = 'D:\\MAS project\\Video\\20240626_112320.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

bg_subtractor = cv2.createBackgroundSubtractorMOG2()

frame_count = 0
stitches = []

while True:
    
    ret, frame = cap.read()

   
    if not ret:
        break

   
    fg_mask = bg_subtractor.apply(frame)

   
    fg_mask = cv2.erode(fg_mask, None, iterations=2)

    
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

   
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   
    print(f"Frame {frame_count}: {len(contours)} contours found")

    
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 500]

    
    for contour in filtered_contours:
        print(f"Contour area: {cv2.contourArea(contour)}")

    
    if len(filtered_contours) > 0:
        
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

       
        print(f"Bounding rectangle: x={x}, y={y}, w={w}, h={h}")

        
        if 100 < x < 300 and 100 < y < 300:
            
            stitches.append(frame_count)

            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

   
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fg_mask)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

   
    frame_count += 1


print(stitches)


cap.release()
cv2.destroyAllWindows()
