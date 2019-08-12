**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./submission/uncorr.png "Uncorrected"
[image2]: ./submission/corr.png "Corrected"
[image3]: ./submission/test_image.jpg "Test Image"
[image4]: ./submission/undistorted.jpg "Undistorted Test Image"
[image5]: ./submission/combined_filts.jpg "Binary Example"
[image6]: ./submission/per.jpg "Warp Example"
[image7]: ./submission/search_around.jpg "Search Around"
[image8]: ./submission/res_test.jpg "Output"

[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]
![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Thresholding values are in cell 83 and done using functions (xgradient and colorthreshold) then combined by combinefilters.  Here's an example of my output for this step!

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective()`, which appears in cell 16.  The `perspective()` function takes as inputs an image (`img`) and return the warped image along with the inverse matrix. I chose the hardcode the source and destination points in the following manner:

```python
    src = np.array([[combined_binary.shape[1],combined_binary.shape[0]-10], # col, row
                    [0,combined_binary.shape[0]-10],
                    [546,460],
                    [732,460]],
                   np.float32)
    dst = np.array([[combined_binary.shape[1],combined_binary.shape[0]], # col, row
                    [0,combined_binary.shape[0]],
                    [0,0],
                    [combined_binary.shape[1],0]],
                   np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 1200, 710      | 1200, 720        | 
| 0, 710      | 0, 720      |
| 546, 460     | 0, 0      |
| 732, 460      | 1200, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in cells 27 and 31. Basically roc is done same way in the quiz. Vehicle offset is the difference between the middle of image and delta of the two lane lines, all in real measurements domain.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 83 where i ran the pipeline with a test image.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

More thresholding techniques should be involved. May be incorporate different color channels or direction gradients. Not expected to work perfectly in shadows, but if i have time i would create tool to modify all the thresholds and which channels are being used instantly to find the perfect combination. Not sure about how efficient is the pixel finding algorithm in terms of time and memory needed. 
