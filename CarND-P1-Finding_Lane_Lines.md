# **P1 - Finding Lane Lines on the Road** 

### Description

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm. This project detect lane lines in images using Python and OpenCV.  I used the tools that I learned about in the lesson (Computer Vision Fundamentals) to identify lane lines on the road. You can run the pipeline on a series of individual images, and video stream (really just a series of images) in the same script. 

---

### Used Methods

The tools that I used are color space (HSV and HLS), region of interest selection, gray scaling, Gaussian smoothing, Canny Edge Detection and Hough Transform line detection. To achieve the goal was piece together a pipeline to detect the line segments in the image/video, averaging/extrapolating them and draw them onto the image/video for display. 

---

### How to run
To run the pipeline just run in a prompt the command:

```clear && python2 CarND-P1-Finding_Lane_Lines.py```

You can change or specified some process variables as:	    
	folder_results       = "./results"         # Results/output folder where result will be saved
    	folder_dir_image = "./test_images" # folder with images to process 
    	folder_dir_video  = "./test_videos"  # folder with videos to process 
    	Tune_ranges = False # Enable/Disable parameters tuning (Shows a window to tune color space model values for thresholding)
    	Save_results = True  # Enable/Disable results saving (Write images and videos in folder_results)

Tested on: python 2.7 (3.X should work), OpenCV 3.0.0 (Higher version should work), UBUNTU 16.04.

---

### Code Description

I decided to explore my own methods and write all functions from scratch, so no given function was used or modified for this project. The code “CarND-P1-Finding_Lane_Lines.py” is well documented, but here I’ll resume the process to achieve the goal in this project:

**Step 0**: I decided to use a color space (HSV) to get a binary image with our objects of interest (White lane lines and Yellow lane lines), but setting a maximum and a minimum value for each parameter H (Hue), S (Saturation), and V (Value) and then compile, see result, adjust and try again is bothersome, so I coded a simple tuner for this. Using the function “color_range_tunner”  you can load stetted parameters and set new values for a image. So, I tuned to color ranges for White lines and Yellow lines (white_conf_hsv.npz and yellow_conf_hsv.npz). If you decide don't tune up any parameter, this function loads parameters and return their values from npz files. It’s possible to tune parameters for a different color space instead of HSV like HLS o others supported by OpenCV (I just played with the HSV space).

<img src="/writeup_files/HoughLinesP_Heuricstic.png" alt="drawing" width="320"/>
Figure 1 - Original image 

<img src="/writeup_files/tuner_window.png" alt="drawing" width="320"/>
Figure 2 - Color space tuner window 

With these setted parameters now let’s see the main function which finds and returns right and left lane line:

find_lanelines(img_src, COLOR_TRESH_MIN, COLOR_TRESH_MAX, COLOR_MODEL, HOZ_TRESH = 0.6, FILT_KERN = 5)   

**Input arguments**:
	img_src: `cv2.math` input image to find and approximate lane lines
	COLOR_TRESH_MIN: `list` Minimum parameters to color thresholding
	COLOR_TRESH_MAX: `list` Maximum parameters to color thresholding
	COLOR_MODEL: `list` Color space in cv2 interface to color thresholding
	VERT_TRESH: `float` Normalized value to ignore vertical image range
	FILT_KERN: `int` (odd) size/kernel of filter (Bilateral)


**Setp 1**: Smooth the image with a bilateral filter with a kernel size given by “FILT_KERN”. This kind of filter can reduce unwanted noise very well while keeping edges fairly sharp (but it is very slow compared to most filters).

<img src="/writeup_files/image_filtered.png" alt="drawing" width="320"/>
Figure 3 - Smoothed Image with bilateral filter

**Setp 2**: Get a binary mask from every color space tunned (COLOR_TRESH_MIN, COLOR_TRESH_MAX, COLOR_MODEL), and then apply a logical operator (OR) two combine all mask in just one and get a binary image/mask, .

<img src="/writeup_files/binary_mask.png" alt="drawing" width="320"/>
Figure 4 - Binary image from color thresholding

**Setp 3**: Then Hough Line Transform is applied, no matter what parameter values are specified here since our image is a binary mask : 

<img src="/writeup_files/mask_canny.png" alt="drawing" width="320"/>
Figure 5 - Canny edge detection algorithm

<img src="/writeup_files/HoughLinesP.png" alt="drawing" width="320"/>
Figure 6 - Probabilistic Hough Line algorithm

<img src="/writeup_files/HoughLinesP_Heuricstic.png" alt="drawing" width="320"/>
Figure 7 - Hough Lines filtered and assigned with heuristic"

**Returns**:
	Lm: `float`  linear regression slope of left lane line
	Lb: `float`  linear regression y-intercept of left lane line
	Rm: `float`  linear regression slope of right lane line
	Rb: `float`  linear regression y-intercept of right lane line
	Left_Lines: `list` list of left lines with which left lane line was calculated
	Right_Lines: `list` list of left lines with which right lane line was calculated

<img src="/writeup_files/result.png" alt="drawing" width="320"/>
Figure 8 - Result of lane lines finding process"

---

### Pontential Shortcommings

---

### Possible Improvements

---

Date: 02/18/2019
Programmer: John A. Betancourt G.
Phone: +57 (311) 813 7206 / +57 (350) 283 51 22
Mail: john.betancourt93@gmail.com / john@kiwicampus.com
Web: www.linkedin.com/in/jhon-alberto-betancourt-gonzalez-345557129





