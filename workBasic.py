import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize a flag to track if we are in "TURN LOW" state
low_beam = False
# Timer to ensure we display "STAY HIGH" after the light disappears
no_light_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to smooth out the image (helps reduce noise)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Thresholding with higher threshold value to get stronger bright spots
    _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)         # Adjust threshold value as needed upto 245 only.

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    high_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 1500:  # Filter for medium-sized light sources (headlights)

            # Get bounding box of detected contours
            x, y, w, h = cv2.boundingRect(cnt)

            # Only process the lower half of the image (potential headlights)
            if y + h > frame.shape[0] * 0.4:  # If the object is in the lower half
                high_detected = True
                # Draw rectangle around detected bright spot
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # If a light source (vehicle light) is detected
    if high_detected:
        low_beam = True
        no_light_counter = 0  # Reset the counter when light is detected
        cv2.putText(frame, "TURN LOW", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    else:
        # If the light is not detected, increase the counter
        if low_beam:
            no_light_counter += 1
            print(f"No light detected, counter: {no_light_counter}")  # Debugging

            # Once no light is detected for a set number of frames, switch to "STAY HIGH"
            if no_light_counter > 30:  # Adjust the number for your desired timeout (frames)
                low_beam = False
                print("No light for 30 frames, switching to STAY HIGH")  # Debugging
                cv2.putText(frame, "STAY HIGH", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            else:
                # Show "TURN LOW" for a while after the light is gone
                cv2.putText(frame, "TURN LOW", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        else:
            # This will be the case when light is not detected and we're in "STAY HIGH" state
            cv2.putText(frame, "STAY HIGH", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    # Show the frame
    cv2.imshow('Frame', frame)

    # Break with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()
