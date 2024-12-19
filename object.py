import RPi.GPIO as GPIO
from RPLCD.gpio import CharLCD
import cv2
import time

# Set up GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# LCD Setup
lcd = CharLCD(
    numbering_mode=GPIO.BCM,
    cols=16, rows=2,
    pin_rs=21, pin_e=20,
    pins_data=[16, 12, 25, 24]
)

# Set up LED pins
RED_LED_PIN = 22
GREEN_LED_PIN = 27
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)

# Initialize the camera and object variables
cap = cv2.VideoCapture(0)
object_count = 0
object_length = 0.0
object_width = 0.0
line_position = 300  # Adjust based on your frame height
last_update_time = time.time()

# Main loop for object detection and display update
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame (convert to grayscale, blur, and threshold)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Detect the main object
        main_object = max(contours, key=cv2.contourArea, default=None)
        if main_object is not None:
            x, y, w, h = cv2.boundingRect(main_object)
            if 200 < cv2.contourArea(main_object) < 10000 and 0.2 < float(w) / h < 5.0:
                current_time = time.time()
                if current_time - last_update_time >= 2:
                    last_update_time = current_time
                    if y + h // 2 > line_position:
                        object_count += 1
                        object_length = h / 10.0
                        object_width = w / 10.0
                    GPIO.output(RED_LED_PIN, GPIO.LOW)
                    GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
                else:
                    GPIO.output(GREEN_LED_PIN, GPIO.LOW)
                    GPIO.output(RED_LED_PIN, GPIO.HIGH)

                # Update LCD display with count, length, and width
                lcd.clear()
                lcd.write_string(f"Count: {object_count}")
                lcd.cursor_pos = (1, 0)
                lcd.write_string(f"Len:{object_length:.1f} W:{object_width:.1f}")

            # Draw rectangle on main object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw the line on the frame and display text
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Length: {object_length:.1f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Width: {object_width:.1f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with overlaid text
        cv2.imshow('Object Detection', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program stopped")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    lcd.clear()
    GPIO.cleanup()
