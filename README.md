# Digital-Analog-Meter-Reader-with-Wireless-Live-Streaming-and-Photo-Capturing-Capabilities
The attached file 'main.py' could be run at Visual Studio (VS) Code, Thorny, Python IDLE, etc. Make sure you install the library below at the latest version before compiling the code:
a. cv2
b. numpy
c. time
d. sys
e. tkinker
f. matplotlib

[Description]
**Live Stream	**
1.	Access CameraWebServer through Wi-Fi credentials
In this and subsequent section, ESP32-CAM board is used to includes a camera module that can be used to capture video footage. The outcome for this section is the configuration Wi-Fi settings. Referring to Appendix A, user would need to define the WiFi SSID and password that will be connected to ESP32-CAM. Once the code was uploaded into the board, the board will automatically look for the network once it is powered on every time. 
2.	IP address of camera
Extract the IP address from the Serial Monitor after running CameraWebServer.ino.
3.	Enable video capturing
With the web server and camera module configured, user can now start streaming video over the internet. To start the frame capturing, cv2.VideoCapture(‘IP address’) and cv2.imshow() function from OpenCV library is used. To exit the video streaming, cap.release() function will be implemented.

**Digital Gauge Reader	**
1.	Rectangle Detection
It is essential for the robot to detect the presence of Region of Interest (ROI) from the digital gauges from Appendix C. Commonly, ROI of digital gauges are bounded with rectangular contour. First, the video is being grayscale and applied with Gaussian blur and threshold to reduce white noise and increase contrast. Followed by Canny edge detector to detect the edges of objects inside the frame. Next, using the function cv2.findCountours() and setting cv2.contourArea sufficiently large, so robot will be able to detect the largest rectangle from the frame.
2.	ROI Extraction
As the algorithm successfully recognized the ROI with the area more than 1000, the system will crop the frame with the detected width and height of the meter screen.
3.	ROI Processing
The cropped frame will be carried for media pre-processing with grayscale using cv2.cvtColor(), application of Gaussian blur and inversion on the frame, application of contrast to the frame through CLAHE (Contrast Limited Adaptive Histogram Equalization) and cv2.adaptiveThreshold(). Lastly, cv2.morphologyEx() helps to fill small holes and gaps in the foreground (white regions) of the binary image in a bid to remove noise and small objects from the binary image.
4.	Digital Look Up
In the custom functions called ‘recognize_digits_area_method’ and ‘recognize_digits_line_method’, these functions segmentation and pattern matching approach based on dividing the ROI into different segments and evaluating each segment for the presence of a digit. The pattern of "on" and "off" segments is compared to predefined patterns in the DIGITS_LOOKUP dictionary. 
5.	Display Result
If a match is found, the corresponding digit is appended to the digits list. Additionally, if a certain region in the ROI has a high density of non-zero pixels, indicating the presence of a dot, a decimal point is appended to the digits list. The digits list will be passed to cv2.putText() function to perform printing on the live window using text annotations.

**Analogue Gauge Reader	**
1.	Gauge Calibration
The calibration parameters such as min_angle, max_angle, min_value, and max_value is set to define the range and units of the gauge. These parameters are essential for mapping the detected angle to a meaningful value.   
2.	Circle Detection
Circle detection is performed using the Hough Circle Transform. The cv2.HoughCircles() function is used to detect circles in the frame. The detected circles are then averaged using the avg_circles function to obtain a more accurate representation of the gauge center and radius. This is essential for the system to trigger the subsequent stages to save the computational resource.
3.	Needle Detection
Needle detection is carried out using the Canny edge detection and Hough line transform. The system would find the longest line, which represents the needle, and draws it on the frame.
4.	Angle Calculation
The system will calculate the angle of the needle with respect to the center of the gauge. It uses the coordinates of the needle endpoints and computes the angle using trigonometric functions.
5.	Display Result
The final angle is mapped to a value within the specified range (min_value to max_value). The result, including the value and units, is displayed on the frame using text annotations.
