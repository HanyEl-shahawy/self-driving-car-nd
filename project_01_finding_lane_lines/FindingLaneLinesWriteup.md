# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 
1- Convert to grayscale
2- Blur with kernel size of 7*7
3- Apply canny edge detection algorithm with threshold values of 30 and 120
4- Extract region of interest which is triangle with lower vertices shifted 100 pixels from both sides and apex value at exactly center of image
5- Transform edges image to Hough space with rho=1, theta=pi/180, threshold=27 min_line_length=18 and max_line_gap=2
6- Modified draw_lines(); loops on all detected lines and label points as left lane or right lane depending on line slope.
Fit a line between points for each line to get line equation.
Evaluate (x,y) for lane line at bottom of image and at top point of lane line.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what happen at curved lanes or different light conditions. 


### 3. Suggest possible improvements to your pipeline

Use 3rd degree polynomial for line fitting.
