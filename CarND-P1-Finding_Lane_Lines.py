# =============================================================================
"""
Code Information:
    Date:         02/18/2019
	Programmer: John A. Betancourt G.
	Phone: +57 (311) 813 7206 / +57 (350) 283 51 22
	Mail: john.betancourt93@gmail.com / john@kiwicampus.com
    Web: www.linkedin.com/in/jhon-alberto-betancourt-gonzalez-345557129

Description: Project 1 - Udacity - self driving cars Nanodegree
    Finding Lane Lines on the Road

Tested on: python 2.7, OpenCV 3.0.0, UBUNTU 16.04
"""

# =============================================================================
# LIBRARIES AND DEPENDENCIES - LIBRARIES AND DEPENDENCIES - LIBRARIES AND DEPEN
# =============================================================================
#importing useful packages
from __future__ import print_function, division
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# =============================================================================
# LANE LINES CLASS - LANE LINES CLASS - LANE LINES CLASS - LANE LINES CLASS - L
# =============================================================================
class LaneLine:

    def __init__(self, lane_side = "Unknow"):

        """  Lane line class 
        Args:
            lane_side: `string`  lane line label or lane side (right or left)
        Returns:
        """

        self.regression = {"m": None, "b": None} # Current values for linear regression
        self.lane_side = lane_side  # lane line label or lane side (right or left)
        self.regression_m = [] # list of y-intercept of linear regression in time
        self.regression_b = [] # list of slope of the line of linear regression in time
        self.lost_frames = 0 # For how many frames the line is lost
        self.lost = True # State of lane line
        self.lines = [] # Current independent lines for regression

    def associate_regresion(self, m, b, lines):

        """  Associate the new values of linear regression to lane line and 
             calculates a new regression with these values
        Args:
            lane_side: `string`  lane line label or lane side (right or left)
        Returns:
        """

        # Asign lines
        self.lines = lines

        # If there's no linear regression values then do nothing
        if m is None or b is None:
            self.lost = True        # Associate as lost lane line
            self.lost_frames += 1   # Increment the number of frames where lane line is lost
            return
        
        self.lost_frames = 0 # Associate as no lost lane line
        self.lost = False # reset the number of frames where lane line is lost

        # If there's more than 'N' associations delete the first in history
        if len(self.regression_m) > 10:
            self.regression_m.pop(0)
            self.regression_b.pop(0)

        # Add new regretion values 
        self.regression_m.append(m)
        self.regression_b.append(b)

        # Calculate the mean value for each linear regression parameter
        self.regression["m"] = np.mean(self.regression_m)
        self.regression["b"] = np.mean(self.regression_b)

    def __str__(self):
        str2print = "Lane side: {}".format(self.lane_side) +\
                    "Lost: {} ({})".format(len(self.lost), self.lost_frames) +\
                    "Lines: {}".format(len(self.lines)) +\
                    "m: {}".format(round(self.regression["m"],2)) +\
                    "b: {}".format(round(self.regression["b"],2))
        return str2print

# =============================================================================
# FUNCTIONS - FUNCTIONS - FUNCTIONS - FUNCTIONS - FUNCTIONS - FUNCTIONS - FUNCT
# =============================================================================
def print_list_text(img_src, str_list, origin = (0, 0), color = (0, 255, 255), thickness = 2, fontScale = 0.45):

    """  prints text list in cool way
    Args:
        img_src: `cv2.math` input image to draw text
        str_list: `list` list with text for each row
        origin: `tuple` (X, Y) coordinates to start drawings text vertically
        color: `tuple` (R, G, B) color values of text to print
        thickness: `int` thickness of text to print
        fontScale: `float` font scale of text to print
    Returns:
        img_src: `cv2.math` input image with text drawn
    """

    y_space = 20

    for idx, strprint in enumerate(str_list):
        cv2.putText(img = img_src,
                    text = strprint,
                    org = (origin[0], origin[1] + (y_space * idx)),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = fontScale, 
                    color = (0, 0, 0), 
                    thickness = thickness+3, 
                    lineType = cv2.LINE_AA)
        cv2.putText(img = img_src,
                    text = strprint,
                    org = (origin[0], origin[1] + (y_space * idx)),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = fontScale, 
                    color = color, 
                    thickness = thickness, 
                    lineType = cv2.LINE_AA)

    return img_src

def nothing(x): pass

def color_range_tunner(img_src, tune = True, conf_file_path = "", space_model = cv2.COLOR_BGR2HSV):

    """ creates a window to tune with input image 'img_src' parameters of a
        color space model, then saves the configuration parameters in a npz 
        file to used later, function also returns parameters if tune mode is
        off
    Args:
        img_src: `cv2.math` input image to tune parameters of color space model 
        tune: `boolean` Enable/Disable tuner mode
        conf_file_path: `string` npz configuration file to save or load parameters
        space_model: `cv2.flag` Color space for opencv interface
    Returns:
        Arg1Min: `int` Minimum value of first argument in color model (HSV = H / HLS = H)
        Arg2Min: `int` Minimum value of second argument in color model (HSV = S / HLS = L)
        Arg3Min: `int` Minimum value of third argument in color model (HSV = V / HLS = S)
        Arg1Max: `int` Maximum value of first argument in color model (HSV = H / HLS = H)
        Arg2Max: `int` Maximum value of second argument in color model (HSV = S / HLS = L)
        Arg3Max: `int` Maximum value of third argument in color model (HSV = V / HLS = S)
    """

    # First values assignation
    Arg1Min = Arg2Min = Arg3Min = Arg1Max = Arg2Max = Arg3Max = 0

    # Read saved configuration
    if os.path.isfile(conf_file_path):
        npzfile = np.load(conf_file_path)
        Arg1Min = npzfile["Arg1Min"]; Arg2Min = npzfile["Arg2Min"]; Arg3Min = npzfile["Arg3Min"] 
        Arg1Max = npzfile["Arg1Max"]; Arg2Max = npzfile["Arg2Max"]; Arg3Max = npzfile["Arg3Max"]  
    else: 
        print(bcolors.WARNING + "\tWarning: No configuration file" + bcolors.ENDC)

    # Return parameters if not tune
    if not tune:
        return Arg1Min, Arg2Min, Arg3Min, Arg1Max, Arg2Max, Arg3Max

    print("Tuner mode")
    print("\tType 'Q' to quit")

    if space_model == cv2.COLOR_BGR2HSV:
        win_name = "HSV_tunner"
        args = ["Hmin", "Hmax", "Smin", "Smax", "Vmin", "Vmax"]
    elif space_model == cv2.COLOR_BGR2HLS:
        win_name = "HLS_tunner"
        args = ["Hmin", "Hmax", "Lmin", "Lmax", "Smin", "Smax"]
    else:
        return Arg1Min, Arg2Min, Arg3Min, Arg1Max, Arg2Max, Arg3Max

    # Create tuner window
    cv2.namedWindow(win_name)

    # Set track bars to window
    cv2.createTrackbar(args[0], win_name ,Arg1Min, 255, nothing)
    cv2.createTrackbar(args[1], win_name ,Arg1Max, 255, nothing)
    cv2.createTrackbar(args[2], win_name ,Arg2Min, 255, nothing)
    cv2.createTrackbar(args[3], win_name ,Arg2Max, 255, nothing)
    cv2.createTrackbar(args[4], win_name ,Arg3Min, 255, nothing)
    cv2.createTrackbar(args[5], win_name ,Arg3Max, 255, nothing)

    # Create copy of input image
    img_aux = img_src.copy()

    uinput = '_'
    while not (uinput == ord('q') or uinput == ord('Q')):

        # Get trackbars position
        Arg1Min = cv2.getTrackbarPos(args[0], win_name)
        Arg1Max = cv2.getTrackbarPos(args[1], win_name)
        Arg2Min = cv2.getTrackbarPos(args[2], win_name)
        Arg2Max = cv2.getTrackbarPos(args[3], win_name)
        Arg3Min = cv2.getTrackbarPos(args[4], win_name)
        Arg3Max = cv2.getTrackbarPos(args[5], win_name)

        # Set umbrals
        COLOR_LANE_LINE_MIN = np.array([Arg1Min, Arg2Min, Arg3Min],np.uint8)     
        COLOR_LANE_LINE_MAX = np.array([Arg1Max, Arg2Max, Arg3Max],np.uint8)

        # Convert and thresh 
        img_color = cv2.cvtColor(img_aux, space_model)
        mask = cv2.inRange(img_color, COLOR_LANE_LINE_MIN, COLOR_LANE_LINE_MAX)

        # Show result
        result_img = np.concatenate((img_aux, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), axis = 1)
        result_img = cv2.resize(result_img, (int(result_img.shape[1]*0.5), int(result_img.shape[0]*0.5)))
        cv2.putText(img = result_img, text = conf_file_path, org = (10, 20), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, thickness = 3, color = (0, 0, 0))
        cv2.putText(img = result_img, text = conf_file_path, org = (10, 20), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, thickness = 1, color = (0, 255, 255))
        cv2.imshow(win_name, result_img)
        
        # Read user input
        uinput = cv2.waitKey(30) & 0xFF

    # Destroy created windows
    cv2.destroyAllWindows()

    # Save configuration in file
    np.savez_compressed(
        conf_file_path, 
        Arg1Min  = Arg1Min, Arg2Min = Arg2Min, Arg3Min = Arg3Min, 
        Arg1Max  = Arg1Max, Arg2Max = Arg2Max, Arg3Max = Arg3Max)
    print("\tNew {} configuration saved".format(conf_file_path))
    
    # Return tunned parameters
    return Arg1Min, Arg2Min, Arg3Min, Arg1Max, Arg2Max, Arg3Max

def line_intersection(line1, line2):

    """  Finds the intersection coordinate between two lines
    Args:
        line1: `tuple` line 1 to calculate intersection coordinate (X, Y) [pix]
        line2: `tuple` line 2 to calculate intersection coordinate (X, Y) [pix]
    Returns:
        inter_coord: `tuple` intersection coordinate between line 1 and line 2
    """

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) 

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)

     # if lines do not intersect
    if div == 0:
       return 0, 0
       
    # Calculates intersection cord
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    inter_coord = int(round(x)), int(round(y))

    # Return X and Y cords of intersection
    return inter_coord

def find_lanelines(img_src, COLOR_TRESH_MIN, COLOR_TRESH_MAX, COLOR_MODEL, VERT_TRESH = 0.6, FILT_KERN = 5):

    """ Finds a simple linear regression canny edge detection for each lane line (Right and Left)
    Args:
        img_src: `cv2.math` input image to find and approximate lane lines
        COLOR_TRESH_MIN: `list` Minimum parameters to color thresholding
        COLOR_TRESH_MAX: `list` Maximum parameters to color thresholding
        COLOR_MODEL: `list` Color space in cv2 interface to color thresholding
        VERT_TRESH: `float` Normalized value to ignore vertical image values
        FILT_KERN: `int` (odd) size/kernel of filter (Bilateral)
    Returns:
        Lm: `float`  linear regression slope of left lane line
        Lb: `float`  linear regression y-intercept of left lane line
        Rm: `float`  linear regression slope of right lane line
        Rb: `float`  linear regression y-intercept of right lane line
        Left_Lines: `list` list of left lines with which left lane line was calculated
        Right_Lines: `list` list of left lines with which right lane line was calculated
    """

    # -------------------------------------------------------------------------
    # GET BINARY MASK - GET BINARY MASK - GET BINARY MASK - GET BINARY MASK - G

    # Apply convolution filter to smooth the image
    kernel = np.ones((FILT_KERN, FILT_KERN),np.float32)/(FILT_KERN**2)
    # img_filt = cv2.filter2D(img_src, -1, kernel)
    img_filt = cv2.bilateralFilter(src = img_src, d = FILT_KERN, sigmaColor = 75, sigmaSpace = 75)
    # cv2.imshow("image_filtered", img_filt)

    # Crete a mask/ Binary image to find the lane lines
    mask_1 = np.zeros((img_filt.shape[0], img_filt.shape[1], 1), dtype=np.uint8)
    for idx in range(0, len(COLOR_MODEL)):
        # Convert images to hsv (hue, saturation, value) color space
        img_tresh = cv2.cvtColor(src = img_filt.copy(), code = COLOR_MODEL[idx])
        img_tresh[0:int(img_tresh.shape[0]*VERT_TRESH),:] = [0, 0, 0]

        # Thresh by color for yellow lane lines
        img_tresh = cv2.inRange(
            src = img_tresh, 
            lowerb = COLOR_TRESH_MIN[idx], 
            upperb = COLOR_TRESH_MAX[idx])

        # Conbine masks with OR operation
        mask_1 = cv2.bitwise_or(mask_1, img_tresh)
    # cv2.imshow("binary_mask", mask_1)

    # Get canny image
    mask_canny = cv2.Canny(image = mask_1, threshold1 = 0, threshold2 = 255)
    # cv2.imshow("mask_canny", mask_canny)

    # Convert to RGB space to draw results 
    mask = cv2.cvtColor(mask_canny, cv2.COLOR_GRAY2BGR)

    # -------------------------------------------------------------------------
    # IDENTIFY RIGHT AND LEFT LANE LINES - IDENTIFY RIGHT AND LEFT LANE LINES -

    # Get lane lines
    lane_lines = cv2.HoughLinesP(
            image = mask_canny, 
            rho = 0.2, 
            theta = np.pi/180.,
            lines = 100, 
            threshold = 10,     # (10) The minimum number of intersections to detect a line 
            minLineLength = 10, # (10) The minimum number of points that can form a line. 
                                # Lines with less than this number of points are disregarded.
            maxLineGap = 5)     # (5) The maximum gap between two points to be considered in the same line.

    # to draw and show results
    # for line in lane_lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img_src, (x1, y1), (x2, y2),(0, 0, 255), 2)
    # cv2.imshow("HoughLinesP", img_src)

    # Declaration of useful variables
    Left_Lines = []; Right_Lines = []
    bottom_line = ((0, img_src.shape[0]), (img_src.shape[1], img_src.shape[0]))
    Lllx = []; Llly = []; Rllx = []; Rlly = []
    
    if lane_lines is None:
        return None, None, None, None, None, None

    # Apply some heuristics to remove some detected lane lines
    for line in lane_lines:
        for x1, y1, x2, y2 in line:
            angle = math.atan2(y2 - y1, x2 - x1)*(180.0 / math.pi)
            interc = line_intersection(bottom_line, ((x1, y1), (x2, y2)))

            # Conditions for left lines
            COND1 = angle < -10 and angle > -80
            COND2 = x1 < img_src.shape[1]*0.5 or x2 < img_src.shape[1]*0.5
            COND3 = interc[0] > 0

            # Conditions for right lines
            COND4 = int(angle) > 10
            COND5 = x1 > img_src.shape[1]*0.5 or x2 > img_src.shape[1]*0.5
            COND6 = interc[0] > 0 and interc[0] < img_src.shape[1]

            # For left lines
            if COND1 and COND2 and COND3:
                Left_Lines.append(line)
                cv2.line(mask, (x1, y1), (x2, y2),(255, 200, 0), 2)
                Lllx.append(x1); Lllx.append(x2); 
                Llly.append(y1); Llly.append(y2); 

            # For right lines
            elif COND4 and COND5 and COND6:
                Right_Lines.append(line)
                cv2.line(mask, (x1, y1), (x2, y2),(77, 195, 255), 2)
                Rllx.append(x1); Rllx.append(x2); 
                Rlly.append(y1); Rlly.append(y2); 

    # Find linear regresion to aproximate each lane line    
    Lm = None; Lb = None
    Rm = None; Rb = None

    # Calculate simple linear regression for left lane line
    if len(Llly) and len(Lllx):
        Lm, Lb = np.polyfit(x = Lllx, y = Llly, deg = 1, rcond=None, full=False, w=None, cov=False)

    # Calculate simple linear regretion for right lane line
    if len(Rllx) and len(Rlly):
        Rm, Rb = np.polyfit(x = Rllx, y = Rlly, deg = 1, rcond=None, full=False, w=None, cov=False)

    # # to draw and show results
    # for line in Right_Lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img_src, (x1, y1), (x2, y2),(220, 280, 0), 2)
    # for line in Left_Lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img_src, (x1, y1), (x2, y2),(255, 0, 255), 2)
    # cv2.imshow("HoughLinesP_Heuricstic", img_src)

    # Return results
    return Lm, Lb, Rm, Rb, Left_Lines, Right_Lines

def draw_lanelines(img_src, Right_Lane_Line, Left_Lane_Line, VERT_TRESH = 0.6, draw_lines = True, draw_regression = True, draw_info = True):

    """  Draws lane lines in image
    Args:
        img_src: `cv2.math` input image to draw lane lines
        Right_Lane_Line: `LaneLine` Right lane line
        Left_Lane_Line: `LaneLine` Right lane line
        VERT_TRESH: `float` normalized vertical value to start printing lines
        draw_lines: `boolean` input Enable/Disable line printings
        draw_regression: `boolean` Enable/Disable linear regression printings
        draw_info: `boolean` Enable/Disable information printings
    Returns:
        img_src: `cv2.math` input image with lane lines drawn
    """



    # Create a copy of input image
    img_foreground = img_src.copy()

    # If draw line regressions 
    if draw_regression:

        Ly1 = Ry1 = int(img_src.shape[0]*VERT_TRESH)
        Ly2 = Ry2 = img_src.shape[0]

        Rm = Right_Lane_Line.regression["m"]
        Rb = Right_Lane_Line.regression["b"]
        Lm = Left_Lane_Line.regression["m"]
        Lb = Left_Lane_Line.regression["b"]

        line_thickness = 3

        if Lm is not None and Lb is not None:
            Lx1 = int((Ly1 - Lb)/Lm); Lx2 = int((Ly2 - Lb)/Lm)
            line_color = (0, 0, 255) if Left_Lane_Line.lost else (0, 255, 0)
            cv2.line(img_foreground, (Lx1, Ly1), (Lx2, Ly2), line_color, line_thickness)

        if Rm is not None and Rb is not None:
            Rx1 = int((Ry1 - Rb)/Rm); Rx2 = int((Ry2 - Rb)/Rm)
            line_color = (0, 0, 255) if Right_Lane_Line.lost else (0, 255, 0)
            cv2.line(img_foreground, (Rx1, Ry1), (Rx2, Ry2), line_color, line_thickness)

    # If draw individual lines
    if draw_lines:
        if Right_Lane_Line.lines is not None:
            for line in Right_Lane_Line.lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img_foreground, (x1, y1), (x2, y2),(255, 0, 0), 2)
        if Left_Lane_Line.lines is not None:
            for line in Left_Lane_Line.lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img_foreground, (x1, y1), (x2, y2),(255, 0, 0), 2)

    # Overlay image with lines drawn over original image with transparency
    img_src = cv2.addWeighted(
        src2 = img_foreground, 
        src1 = img_src, 
        alpha = 0.7, 
        beta = 0.3, 
        gamma = 0)

     # If print lane lines information in image
    if draw_info:
        if (Left_Lane_Line.lines is not None and  
            Right_Lane_Line.lines is not None and
            Left_Lane_Line.regression["m"] is not None and
            Right_Lane_Line.regression["m"] is not None):
            str_list = ["Left Line:", 
                        "   Line: {}".format(len(Left_Lane_Line.lines)),
                        "   lost: {} ({})".format(Left_Lane_Line.lost, Left_Lane_Line.lost_frames),
                        "   m: {}".format(round(Left_Lane_Line.regression["m"], 2)),
                        "   b: {}".format(round(Left_Lane_Line.regression["b"], 2)),
                        " ",
                        "Right Line:", 
                        "   Line: {}".format(len(Right_Lane_Line.lines)),
                        "   lost: {} ({})".format(Right_Lane_Line.lost, Right_Lane_Line.lost_frames),
                        "   m: {}".format(round(Right_Lane_Line.regression["m"], 2)),
                        "   b: {}".format(round(Right_Lane_Line.regression["b"], 2))]
            
            print_list_text(
                img_src = img_src, 
                str_list = str_list, 
                thickness = 1, 
                origin = (10, 20))

    # Return result
    return img_src

# =============================================================================
# MAIN FUNCTION - MAIN FUNCTION - MAIN FUNCTION - MAIN FUNCTION - MAIN FUNCTION
# =============================================================================
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Config these parameters    
    folder_results   = "./results" # Results/output folder
    folder_dir_image = "./test_images" # folder with images and videos 
    folder_dir_video = "./test_videos"
    Tune_ranges = False # Enable/Disable parameters tuning
    Save_results = False # Enable/Disable results saving

    # -------------------------------------------------------------------------
    img_list = os.listdir(folder_dir_image)
    print(img_list) # Print list of images

    #reading an image
    img = cv2.imread(os.path.join(folder_dir_image, img_list[0]))

    #printing out some stats and plotting
    print('This image is:', type(img), 'with dimensions:', img.shape)

    # -------------------------------------------------------------------------
    # Tuning/Reading color space parameters
    # For color space parameters tunning
    
    YHmin, YSmin, YVmin, YHmax, YSmax, YVmax = color_range_tunner(img_src = img, tune = Tune_ranges, conf_file_path = './yellow_conf_hsv.npz', space_model = cv2.COLOR_BGR2HSV)
    WHmin, WSmin, WVmin, WHmax, WSmax, WVmax = color_range_tunner(img_src = img, tune = Tune_ranges, conf_file_path = './white_conf_hsv.npz', space_model = cv2.COLOR_BGR2HSV)

    # Print tunned parameters
    print("\nYellow:\n\tHmin:{}\t\tHmax:{}\n\tSmin:{}\t\tSmax:{}\n\tVmin:{}\t\tVmax:{}".format(YHmin, YHmax, YSmin, YSmax, YVmin, YVmax))
    print("White :\n\tHmin:{}\t\tHmax:{}\n\tSmin:{}\t\tSmax:{}\n\tVmin:{}\t\tVmax:{}\n".format(WHmin, WSmin, WVmin, WHmax, WSmax, WVmax))

    # Assign minimum and maximum values of color space models 
    YELLOW_COLOR_LANE_LINE_MIN = np.array([YHmin, YSmin, YVmin],np.uint8) # Minimum color values for yellow lines using HSV color space
    YELLOW_COLOR_LANE_LINE_MAX = np.array([YHmax, YSmax, YVmax],np.uint8) # Maximum color values for yellow lines using HSV color space
    WHITE_COLOR_LANE_LINE_MIN = np.array([WHmin, WSmin, WVmin],np.uint8)  # Minimum color values for white lines using HSV color space
    WHITE_COLOR_LANE_LINE_MAX = np.array([WHmax, WSmax, WVmax],np.uint8)  # Maximum color values for white lines using HSV color space

    COLOR_TRESH_MIN = [YELLOW_COLOR_LANE_LINE_MIN, WHITE_COLOR_LANE_LINE_MIN]
    COLOR_TRESH_MAX = [YELLOW_COLOR_LANE_LINE_MAX, WHITE_COLOR_LANE_LINE_MAX]
    COLOR_MODEL = [cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2HSV]

    # -------------------------------------------------------------------------
    # IMAGES - IMAGES - IMAGES - IMAGES - IMAGES - IMAGES - IMAGES - IMAGES - I

    # Read every images in folder 
    for idx in range(0, len(img_list)):

        # Read image
        img = cv2.imread(os.path.join(folder_dir_image, img_list[idx]))

        # Create and initialize Lane lines variables
        Right_Lane_Line = LaneLine(lane_side = "Right")
        Left_Lane_Line = LaneLine(lane_side = "Left")

        # Find lane lines in image
        Lm, Lb, Rm, Rb, Left_Lines, Right_Lines = \
            find_lanelines(
                img_src = img.copy(), 
                COLOR_TRESH_MIN = COLOR_TRESH_MIN, 
                COLOR_TRESH_MAX = COLOR_TRESH_MAX,
                COLOR_MODEL = COLOR_MODEL,  
                VERT_TRESH = 0.6, 
                FILT_KERN = 5)

        # Associate found regression to current lane lines
        Right_Lane_Line.associate_regresion(
            lines = Right_Lines,
            m = Rm, 
            b = Rb)
        Left_Lane_Line.associate_regresion(            
            lines = Left_Lines,
            m = Lm, 
            b = Lb)

        # Draw current lane lines
        img = draw_lanelines(
            img_src = img, 
            Right_Lane_Line = Right_Lane_Line, 
            Left_Lane_Line = Left_Lane_Line, 
            VERT_TRESH = 0.6)

        # Wirte result image
        if Save_results:
            cv2.imwrite(os.path.join(folder_results, img_list[idx]), img)

        # Show results
        cv2.imshow("result", img)
        cv2.waitKey(0)

    # -------------------------------------------------------------------------
    # VIDEOS - VIDEOS - VIDEOS - VIDEOS - VIDEOS - VIDEOS - VIDEOS - VIDEOS - V

    # Variables for videos
    video_list = os.listdir(folder_dir_video)

    for idx in range(0, len(video_list)): 

        # Create and initialize Lane lines variables
        Right_Lane_Line = LaneLine(lane_side = "Right")
        Left_Lane_Line = LaneLine(lane_side = "Left")

        # Variables for video recording
        if Save_results:
            fps = 15. # Frames per second
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec H264, format MP4
            name_1 = os.path.join(folder_results, video_list[idx]) # File name and format
            if 'video_out' in locals(): del video_out
                    
        # Start video capture
        cap = cv2.VideoCapture(os.path.join(folder_dir_video, video_list[idx]))
        reproduce = True

        while(cap.isOpened()):
            
            if reproduce:
            
                # Read frame in video capture
                ret, frame = cap.read()
                if not ret: break

                if not 'video_out' in locals() and Save_results:
                    video_out = cv2.VideoWriter(name_1, fourcc, fps, (frame.shape[1], frame.shape[0]))

                # Find lane lines in image
                Lm, Lb, Rm, Rb, Left_Lines, Right_Lines = \
                    find_lanelines(
                        img_src = frame.copy(), 
                        COLOR_TRESH_MIN = COLOR_TRESH_MIN, 
                        COLOR_TRESH_MAX = COLOR_TRESH_MAX,
                        COLOR_MODEL = COLOR_MODEL,  
                        VERT_TRESH = 0.6, 
                        FILT_KERN = 5)

                # Associate found regression to current lane lines
                Right_Lane_Line.associate_regresion(
                    lines = Right_Lines,
                    m = Rm, 
                    b = Rb)
                Left_Lane_Line.associate_regresion(            
                    lines = Left_Lines,
                    m = Lm, 
                    b = Lb)

                # Draw current lane lines
                frame = draw_lanelines(
                    img_src = frame, 
                    Right_Lane_Line = Right_Lane_Line, 
                    Left_Lane_Line = Left_Lane_Line, 
                    VERT_TRESH = 0.6)

                # Print info in image
                cv2.putText(img = frame, text = video_list[idx], org = (10, int(frame.shape[0]*0.98)), 
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, thickness = 4, color = (0, 0, 0))
                cv2.putText(img = frame, text = video_list[idx], org = (10, int(frame.shape[0]*0.98)), 
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, thickness = 1, color = (255, 255, 0))

                # Show result
                cv2.imshow("result", frame)

                # Write video
                if Save_results: video_out.write(frame) 

            # Wait user input
            user_in = cv2.waitKey(20) 

            if user_in & 0xFF == ord('q') or user_in == ord('Q'): 
                break
            if user_in & 0xFF == ord('a') or user_in == ord('A'): 
                reproduce = not reproduce
            if user_in & 0xFF == ord('t') or user_in == ord('t'): 
                hsv_range_tunner(img_src = frame, tune = True, conf_file_path = './white_conf_hsv.npz')
                hsv_range_tunner(img_src = frame, tune = True, conf_file_path = './yellow_conf_hsv.npz')

    # Destroy video variables
    cap.release()
    cv2.destroyAllWindows()

# =============================================================================
# SORRY FOR MY ENGLISH!