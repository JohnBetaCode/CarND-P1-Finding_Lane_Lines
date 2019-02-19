# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="writeup_files/result.jpg" width="480" alt="Combined Image" />

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

This project detect lane lines in images using Python and OpenCV. OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

In this project, I used the tools that I learned about in the lesson (Computer Vision Fundamentals from udacity self driving cars - Nano Degree program) to identify lane lines on the road. You can run the pipeline on a series of individual images, and video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the outputs look like.

The tools you have here are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection. Other techniques were explored and  they were not presented in the lessons. The goal was piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display. 

For implementation and methods used you can read the writeup.

To run the code just run the next command in promt in root folder:

>>clear && python2 CarND-P1-Finding_Lane_Lines.py

Tested on: python 2.7, OpenCV 3.0.0, UBUNTU 16.04.

