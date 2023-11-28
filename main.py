
# Coder         : Eric Sia Siew Wei
# Background    : Bachelor of Electrical & Electronics Engineering (Hons.)
# University    : Universiti Teknologi PETRONAS (UTP), Malaysia.
# Course        : Integrated System Design Project (ISDP)
# LinkedIn      : https://www.linkedin.com/in/eric-sia-siew-wei/

import numpy as np
import cv2
import time
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

DIGITS_LOOKUP = {
     (1, 1, 1, 1, 1, 1, 0): 0,
     (1, 1, 0, 0, 0, 0, 0): 1,
     (1, 0, 1, 1, 0, 1, 1): 2,
     (1, 1, 1, 0, 0, 1, 1): 3,
     (1, 1, 0, 0, 1, 0, 1): 4,
     (0, 1, 1, 0, 1, 1, 1): 5,
     (0, 1, 1, 1, 1, 1, 1): 6,
     (1, 1, 0, 0, 0, 1, 0): 7,
     (1, 1, 1, 1, 1, 1, 1): 8,
     (1, 1, 1, 0, 1, 1, 1): 9,
     (0, 0, 0, 0, 0, 1, 1): '-'
     }
H_W_Ratio = 1.9
THRESHOLD = 35
arc_tan_theta = 6.0 
crop_y0 = 215
crop_y1 = 470
crop_x0 = 260
crop_x1 = 890
units = "bar"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1  
font_color = (0, 255, 0)  
font_thickness = 2

set = 1
if set == 1:
    fixed_width = 640
    fixed_height = 300
else:
    fixed_width = 320
    fixed_height = 176
min_angle = 45
max_angle = 320
min_value = 0
max_value = 16
units = "bar"
url = 0
#url = 'http://_._._._:81/stream'

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1  
font_color = (0, 255, 0)  
font_thickness = 2

class LiveVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FAME Robot's Vision (FOV)")
        self.frame = tk.Frame(root)
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, width=1500, height=600)
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.create_buttons()
        self.video_capture = cv2.VideoCapture(url)
        self.video_running = False
    def create_buttons(self):
        button_font = ("Arial", 15)
        button_width = 20
        digital_label = tk.Label(self.frame, text="DIGITAL GAUGE READER", font=("Arial", 15))
        digital_label.grid(row=1, column=0, padx=10, pady=(0, 5))  
        digital_button_live = tk.Button(self.frame, text="Live", command=digital_gauge_reader_live, font=button_font, width=button_width)
        digital_button_live.grid(row=2, column=0, padx=10, pady=10)
        digital_button_photo = tk.Button(self.frame, text="Take Photo", command=digital_gauge_reader_taking_photo, font=button_font, width=button_width)
        digital_button_photo.grid(row=3, column=0, padx=10, pady=10)
        analog_label = tk.Label(self.frame, text="ANALOGUE GAUGE READER", font=("Arial", 15))
        analog_label.grid(row=1, column=1, padx=10, pady=(0, 5))  
        analog_button_live = tk.Button(self.frame, text="Live", command=analog_gauge_reader_live, font=button_font, width=button_width)
        analog_button_live.grid(row=2, column=1, padx=10, pady=10)
        analog_button_photo = tk.Button(self.frame, text="Take Photo", command=analog_gauge_reader_taking_photo, font=button_font, width=button_width)
        analog_button_photo.grid(row=3, column=1, padx=10, pady=10)
        setting_label = tk.Label(self.frame, text="CONTROL", font=("Arial", 15))
        setting_label.grid(row=1, column=2, padx=10, pady=(0, 5))       
        quit_button = tk.Button(self.frame, text="Quit", command=self.root.destroy, font=button_font, width=button_width)
        quit_button.grid(row=3, column=2, padx=10, pady=10)
    def toggle_video(self):  
        if self.video_running:
            self.video_running = False
            self.start_button["text"] = "Start"
            self.video_capture.release()  # Release the video capture
        else:
            self.video_running = True
            self.start_button["text"] = "Stop"
            self.video_capture = cv2.VideoCapture(url)  # Reinitialize the video capture
            self.show_video()
    def show_video(self):
        if self.video_running:
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                #frame = cv2.flip(frame, 1)
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(450, 100, image=photo, anchor=tk.NW)
                self.canvas.image = photo
                self.root.after(5, self.show_video)
            else:
                self.toggle_video()
    def run(self):
        button_font = ("Arial", 15)
        button_width = 20
        self.start_button = tk.Button(self.frame, text="Start", command=self.toggle_video, font=button_font, width=button_width)
        self.start_button.grid(row=2, column=2, columnspan=3, pady=10)
        
def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def detect_needle(img):
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is not None:
        longest_line = max(lines, key=lambda x: x[0][0])
        rho = longest_line[0][0]
        theta = longest_line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return [(x1, y1), (x2, y2)]
    return None

def preprocess(img, threshold, show=False, kernel_size=(5, 5)):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.convertScaleAbs(img)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    img = clahe.apply(img)
    dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)

    if show:
        cv2.imshow('equlizeHist', img)
        cv2.imshow('threshold', dst)
    return dst


def helper_extract(one_d_array, threshold=20):
    res = []
    flag = 0
    temp = 0
    for i in range(len(one_d_array)):
        if one_d_array[i] < 12 * 255:
            if flag > threshold:
                start = i - flag
                end = i
                temp = end
                if end - start > 20:
                    res.append((start, end))
            flag = 0
        else:
            flag += 1

    else:
        if flag > threshold:
            start = temp
            end = len(one_d_array)
            if end - start > 50:
                res.append((start, end))
    return res


def find_digits_positions(img, reserved_threshold=20):
    digits_positions = []
    img_array = np.sum(img, axis=0)
    horizon_position = helper_extract(img_array, threshold=reserved_threshold)
    img_array = np.sum(img, axis=1)
    vertical_position = helper_extract(img_array, threshold=reserved_threshold * 4)
    if len(vertical_position) > 1:
        vertical_position = [(vertical_position[0][0], vertical_position[len(vertical_position) - 1][1])]
    for h in horizon_position:
        for v in vertical_position:
            digits_positions.append(list(zip(h, v)))
    return digits_positions


def recognize_digits_area_method(digits_positions, output_img, input_img):
    digits = []
    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = input_img[y0:y1, x0:x1]
        h, w = roi.shape
        suppose_W = max(1, int(h / H_W_Ratio))
        if w < suppose_W / 2:
            x0 = x0 + w - suppose_W
            w = suppose_W
            roi = input_img[y0:y1, x0:x1]
        width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
        dhc = int(width * 0.8)

        small_delta = int(h / arc_tan_theta) // 4
        segments = [
            ((w - width - small_delta, width // 2), (w, (h - dhc) // 2)),
            ((w - width - 2 * small_delta, (h + dhc) // 2), (w - small_delta, h - width // 2)),
            ((width - small_delta, h - width), (w - width - small_delta, h)),
            ((0, (h + dhc) // 2), (width, h - width // 2)),
            ((small_delta, width // 2), (small_delta + width, (h - dhc) // 2)),
            ((small_delta, 0), (w + small_delta, width)),
            ((width - small_delta, (h - dhc) // 2), (w - width - small_delta, (h + dhc) // 2))
        ]
        on = [0] * len(segments)

        for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
            seg_roi = roi[ya:yb, xa:xb]
            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9
            print(total / float(area))
            if total / float(area) > 0.45:
                on[i] = 1
        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
        else:
            digit = '?'
        digits.append(digit)
    return digits


def recognize_digits_line_method(digits_positions, output_img, input_img):
    digits = []
    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = input_img[y0:y1, x0:x1]
        h, w = roi.shape
        suppose_W = max(1, int(h / H_W_Ratio))
        if x1 - x0 < 25 and cv2.countNonZero(roi) / ((y1 - y0) * (x1 - x0)) < 0.2:
            continue
        if w < suppose_W / 2:
            x0 = max(x0 + w - suppose_W, 0)
            roi = input_img[y0:y1, x0:x1]
            w = roi.shape[1]
        center_y = h // 2
        quater_y_1 = h // 4
        quater_y_3 = quater_y_1 * 3
        center_x = w // 2
        line_width = 5  
        width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
        small_delta = int(h / arc_tan_theta) // 4
        segments = [
            ((w - 2 * width, quater_y_1 - line_width), (w, quater_y_1 + line_width)),
            ((w - 2 * width, quater_y_3 - line_width), (w, quater_y_3 + line_width)),
            ((center_x - line_width - small_delta, h - 2 * width), (center_x - small_delta + line_width, h)),
            ((0, quater_y_3 - line_width), (2 * width, quater_y_3 + line_width)),
            ((0, quater_y_1 - line_width), (2 * width, quater_y_1 + line_width)),
            ((center_x - line_width, 0), (center_x + line_width, 2 * width)),
            ((center_x - line_width, center_y - line_width), (center_x + line_width, center_y + line_width)),
        ]
        on = [0] * len(segments)
        for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
            seg_roi = roi[ya:yb, xa:xb]
            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9
            if total / float(area) > 0.25:
                on[i] = 1
        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
        else:
            digit = '?'
        digits.append(digit)
        if cv2.countNonZero(roi[h - int(3 * width / 4):h, w - int(3 * width / 4):w]) / (9. / 16 * width * width) > 0.65:
            digits.append('.')
    return digits

def digital_gauge_reader_live():
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
        ret, thr = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(thr, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4 and cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                if set == 1:
                    roi = frame[y+15 :y + h-32, x+30 :x + w -40]    #Set 1 (7.00)
                else:    
                    roi = frame[y+45 :y + h-15, x+30 :x + w -20]    #Set 2 (13.80)
                if roi.size != 0:
                    roi = cv2.resize(roi, (fixed_width, fixed_height))
                    cv2.imshow('Region of Interest', roi)
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  
                    blur = cv2.GaussianBlur(gray, (7, 7), 0)
                    dst = preprocess(blur, THRESHOLD, False)
                    digits_positions = find_digits_positions(dst)
                    if len(digits_positions) > 0:
                        digits = recognize_digits_line_method(digits_positions, frame, dst)
                        cleaned_digits = [str(digit) if str(digit).isdigit() or str(digit) == '.' else '' for digit in digits]
                        cleaned_string = "".join(cleaned_digits)
                        val_text = ""
                        if any(char.isdigit() or char == '.' for char in cleaned_string):
                            val_text = "Current reading: %s %s" % (cleaned_string, units)
                            print(val_text)
                            cv2.putText(frame, val_text, (10, 40), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                        else:
                            confuse = "-"
                            val_text = "Current reading: %s %s" % (confuse, units)
                            cv2.putText(frame, val_text, (10, 40), font, font_scale, font_color, font_thickness, cv2.LINE_AA)            
                        cv2.imshow('Digital Gauge Reader', frame)

        cv2.imshow('Digital Gauge Reader', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def digital_gauge_reader_taking_photo():
    cap = cv2.VideoCapture(url)
    print("Photo Digital Gauge captured!")
    if not cap.isOpened():
        sys.exit('Video source not found')
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
        ret, thr = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(thr, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4 and cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                if set == 1:
                    roi = frame[y+15 :y + h-32, x+30 :x + w -40]    #Set 1 (7.00)
                else:    
                    roi = frame[y+45 :y + h-15, x+30 :x + w -20]    #Set 2 (13.80)
                if roi.size != 0:
                    roi = cv2.resize(roi, (fixed_width, fixed_height))
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  
                    blur = cv2.GaussianBlur(gray, (7, 7), 0)
                    dst = preprocess(blur, THRESHOLD, False)
                    digits_positions = find_digits_positions(dst)
                    if len(digits_positions) > 0:
                        digits = recognize_digits_line_method(digits_positions, frame, dst)
                        cleaned_digits = [str(digit) if str(digit).isdigit() or str(digit) == '.' else '' for digit in digits]
                        cleaned_string = "".join(cleaned_digits)
                        val_text = ""
                        if any(char.isdigit() or char == '.' for char in cleaned_string):
                            val_text = "Current reading: %s %s" % (cleaned_string, units)
                            print(val_text)
                            cv2.putText(frame, val_text, (10, 40), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                        else:
                            confuse = "-"
                            val_text = "Current reading: %s %s" % (confuse, units)
                            cv2.putText(frame, val_text, (10, 40), font, font_scale, font_color, font_thickness, cv2.LINE_AA)            
        break
    cv2.imwrite('Captured Digital Gauge.jpg', frame)
    cv2.imshow('Captured Digital Gauge.jpg', frame)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def analog_gauge_reader_live():
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        sys.exit('Video source not found')
    while(cap.isOpened()): 
        while True:
            try:
                ret, img = cap.read()
                img = cv2.resize(img, (640, 480))
                detect_needle(img)
                cv2.imshow('Analog Gauge Reader', img)
                img_blur = cv2.GaussianBlur(img, (5,5), 3)
                gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
                height, width = img.shape[:2]
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
                if circles is not None:
                    a, b, c = circles.shape
                    x, y, r = avg_circles(circles, b)
                else:
                    print("No circles found in the image")
                cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)
                separation = 10.0 
                interval = int(360 / separation)
                p1 = np.zeros((interval,2))
                p2 = np.zeros((interval,2))
                p_text = np.zeros((interval,2))
                for i in range(0,interval):
                    for j in range(0,2):
                        if (j%2==0):
                            p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
                        else:
                            p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
                text_offset_x = 10
                text_offset_y = 5
                for i in range(0, interval):
                    for j in range(0, 2):
                        if (j % 2 == 0):
                            p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                            p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
                        else:
                            p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                            p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees
                for i in range(0,interval):
                    cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
                    cv2.putText(img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)
                gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thresh = 175
                maxValue = 255
                th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV);
                minLineLength = 10
                maxLineGap = 0
                lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=0)  # rho is set to 3 to detect more lines, easier to get more then filter them out later
                final_line_list = []
                diff1LowerBound = 0.15
                diff1UpperBound = 0.25
                diff2LowerBound = 0.5
                diff2UpperBound = 1.0
                for i in range(0, len(lines)):
                    for x1, y1, x2, y2 in lines[i]:
                        diff1 = dist_2_pts(x, y, x1, y1) 
                        diff2 = dist_2_pts(x, y, x2, y2)  
                        if (diff1 > diff2):
                            temp = diff1
                            diff1 = diff2
                            diff2 = temp
                        if (((diff1<diff1UpperBound*r) and (diff1>diff1LowerBound*r) and (diff2<diff2UpperBound*r)) and (diff2>diff2LowerBound*r)):
                            line_length = dist_2_pts(x1, y1, x2, y2)
                            final_line_list.append([x1, y1, x2, y2])
                x1 = final_line_list[0][0]
                y1 = final_line_list[0][1]
                x2 = final_line_list[0][2]
                y2 = final_line_list[0][3]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('Analog Gauge Reader', img)
                dist_pt_0 = dist_2_pts(x, y, x1, y1)
                dist_pt_1 = dist_2_pts(x, y, x2, y2)
                if (dist_pt_0 > dist_pt_1):
                    x_angle = x1 - x
                    y_angle = y - y1
                else:
                    x_angle = x2 - x
                    y_angle = y - y2
                res = np.arctan(np.divide(float(y_angle), float(x_angle)))
                res = np.rad2deg(res)
                if x_angle > 0 and y_angle > 0:  
                    final_angle = 270 - res
                elif x_angle < 0 and y_angle > 0: 
                    final_angle = 90 - res
                elif x_angle < 0 and y_angle < 0:  
                    final_angle = 90 - res
                elif x_angle > 0 and y_angle < 0:  
                    final_angle = 270 - res
                else:  
                    final_angle = 0  
                old_min = float(min_angle)
                old_max = float(max_angle)
                new_min = float(min_value)
                new_max = float(max_value)
                old_value = final_angle
                old_range = (old_max - old_min)
                new_range = (new_max - new_min)
                new_value = (((old_value - old_min) * new_range) / old_range) + new_min
                val = new_value
                if (val >=0):
                    val_text = "Current reading: %.2f %s" % (val, units)
                    print (val_text)
                    cv2.putText(img, val_text, (10, 40), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                    cv2.imshow('Analog Gauge Reader', img)
                else:
                    val_text = "Current reading: -" 
                    cv2.putText(img, val_text, (10, 40), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                    cv2.imshow('Analog Gauge Reader', img)
                pressed = cv2.waitKey(1)
                if pressed in [ord('q'), ord('Q')]:
                    break
            except ValueError as ve:
                x = "do nothing"
            except IndexError:
                x = "do nothing"
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Alert ! Camera disconnected")

def analog_gauge_reader_taking_photo():
    cap = cv2.VideoCapture(url)
    print("Photo Analogue Gauge captured!")
    if not cap.isOpened():
        sys.exit('Video source not found')
    while(cap.isOpened()): 
        while True:
            try:
                ret, img = cap.read()
                img = cv2.resize(img, (640, 480))
                detect_needle(img)
                img_blur = cv2.GaussianBlur(img, (5,5), 3)
                gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
                height, width = img.shape[:2]
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
                if circles is not None:
                    a, b, c = circles.shape
                    x, y, r = avg_circles(circles, b)
                else:
                    print("No circles found in the image")
                cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)
                separation = 10.0 
                interval = int(360 / separation)
                p1 = np.zeros((interval,2))
                p2 = np.zeros((interval,2))
                p_text = np.zeros((interval,2))
                for i in range(0,interval):
                    for j in range(0,2):
                        if (j%2==0):
                            p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
                        else:
                            p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
                text_offset_x = 10
                text_offset_y = 5
                for i in range(0, interval):
                    for j in range(0, 2):
                        if (j % 2 == 0):
                            p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                            p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
                        else:
                            p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                            p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees
                for i in range(0,interval):
                    cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
                    cv2.putText(img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)
                gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thresh = 175
                maxValue = 255
                th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV);
                minLineLength = 10
                maxLineGap = 0
                lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=0)  # rho is set to 3 to detect more lines, easier to get more then filter them out later
                final_line_list = []
                diff1LowerBound = 0.15
                diff1UpperBound = 0.25
                diff2LowerBound = 0.5
                diff2UpperBound = 1.0
                for i in range(0, len(lines)):
                    for x1, y1, x2, y2 in lines[i]:
                        diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
                        diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
                        if (diff1 > diff2):
                            temp = diff1
                            diff1 = diff2
                            diff2 = temp
                        if (((diff1<diff1UpperBound*r) and (diff1>diff1LowerBound*r) and (diff2<diff2UpperBound*r)) and (diff2>diff2LowerBound*r)):
                            line_length = dist_2_pts(x1, y1, x2, y2)
                            final_line_list.append([x1, y1, x2, y2])
                x1 = final_line_list[0][0]
                y1 = final_line_list[0][1]
                x2 = final_line_list[0][2]
                y2 = final_line_list[0][3]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                dist_pt_0 = dist_2_pts(x, y, x1, y1)
                dist_pt_1 = dist_2_pts(x, y, x2, y2)
                if (dist_pt_0 > dist_pt_1):
                    x_angle = x1 - x
                    y_angle = y - y1
                else:
                    x_angle = x2 - x
                    y_angle = y - y2
                res = np.arctan(np.divide(float(y_angle), float(x_angle)))
                res = np.rad2deg(res)
                if x_angle > 0 and y_angle > 0:  
                    final_angle = 270 - res
                elif x_angle < 0 and y_angle > 0: 
                    final_angle = 90 - res
                elif x_angle < 0 and y_angle < 0:  
                    final_angle = 90 - res
                elif x_angle > 0 and y_angle < 0:  
                    final_angle = 270 - res
                else:  
                    final_angle = 0  
                old_min = float(min_angle)
                old_max = float(max_angle)
                new_min = float(min_value)
                new_max = float(max_value)
                old_value = final_angle
                old_range = (old_max - old_min)
                new_range = (new_max - new_min)
                new_value = (((old_value - old_min) * new_range) / old_range) + new_min
                val = new_value
                if (val >=0):
                    val_text = "Current reading: %.2f %s" % (val, units)
                    print (val_text)
                    cv2.putText(img, val_text, (10, 40), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                    image = img
                   
                else:
                    val_text = "Current reading: -" 
                    cv2.putText(img, val_text, (10, 40), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                    image = img
           
                cv2.imshow('Analog Gauge Reader', image)
                cv2.imwrite('Captured Analogue Gauge.jpg', image)
                pressed = cv2.waitKey(10000)
                if pressed in [ord('q'), ord('Q')]:
                    break
            except ValueError as ve:
                x = "do nothing"
            except IndexError:
                x = "do nothing"

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Alert ! Camera disconnected")

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveVideoApp(root)
    app.run()
    root.mainloop()

