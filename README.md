# **Advanced Lane Finding Project**

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

[image1]: ./output_images/calibration_chessboard.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary_lanelines_test2.jpg "Binary Example"
[image4]: ./output_images/undist_straight_line.jpg "Unwarped Example"
[image5]: ./output_images/line_pixels_poly_fit_test4.jpg "Fit Visual"
[image6]: ./output_images/lane_detected_test4.jpg "Output"
[image7]: ./output_images/calibration_test1.jpg "Calibrated Test1"
[image8]: ./output_images/bird_view_straight_lines.jpg "Warped image"
[video1]: ./detected_project_video.mp4 "Video"

## Something I'd like you to know before reading my code

- There is another branch called **bumpy** where I fixed the lane detection being slightly off the lane when road is bumpy, to some extent. But that version is less robust to shadows than the master branch. That's the difference between applying moving average which is a filter to the recent fit or not.

- I refactored the Python script so it can be used to process videos and generate resulting videos only. All the code used to generate intermediate images were either removed or commented out to increase code readability.

- In order to make my pipeline function compatible with `moviepy` functions I wrapped up image processing pipeline in another function. This is called closure. It is also how currying is done in Python.

- I implemented the function used to do thresholding gradients but decided not to use it. Because a forum mentor said it was terrible with shadows. The final output video is the one generated without gradients. Looks fine.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function named `get_undist_params()` in the file `./pipeline_helpers.py`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

Just apply the camera calibration matrix and distortion coefficients you obtained from the last step like this

```python
undist = cv2.undistort(img, camMat, distCoeffs, None, camMat)
```
What you get is an image like this:

![alt text][image7]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

the relevant code is in `get_bin_lane_line_img(img_gray, img_bgr)` function in `./pipeline.py` and `combine_color(img_bgr, rgb_thresh=(220, 255), hls_thresh=(90, 255))` in `./pipeline_helpers.py`. I used color thresholding only. Because a forum mentor said gradients are terrible with shadows. But I implementd gradients anyway. The code is in `abs_sobel_thresh(), mag_thresh(), dir_thresh(), combine_grad()` in `./pipeline_helpers.py`.

I used R, L and S channels of a colored image. L channel is used to remove the shadows like tree shadows. S channel is pretty good at identifying colored lines. R channel is good at identifying white lines.

The generated image is like this:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To warp an image I need transformation matrices called `matP` and `matP_inv` and the image from last step as the source image.

The code used to generate `matP` and `matP_inv` is in `get_perspective_mat(camMat, distCoeffs)` in `./pipeline.py`. The key is to get the coordinates for 4 points in the unwarped image which I call `src` and the coordinates for 4 points corresponding to `src` in the warped image which I call `dst`.

As the exercise suggested I used an image with straight lane lines, undistorted the image, eyeballed 4 `src` points. Those 4 points form a rectangle in the warped image so they form 2 pairs and the 2 points in each pair have the same y coordinates. The choice of `dst` is simpler, you just need to make sure they form a rectangle.

My `src` and `dst` are like this:

```python
offset_x = 300
offset_y = 0

upper_left = [578, 462]
upper_right = [704, 462]
lower_left = [212, 719]
lower_right = [1093, 719]

src = np.float([upper_left, upper_right, lower_left, lower_right])
dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x - 1, offset_y],
                      [img_size[0] - offset_x - 1, img_size[1] - offset_y - 1],
                      [offset_x, img_size[1] - offset_y - 1]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 579, 463      | 301, 0        |
| 213, 720      | 301, 720      |
| 1094, 720     | 980, 720      |
| 705, 463      | 980, 0        |

Now we have `matP` and `matP_inv`. by

```python
img_size = (undist.shape[1], undist.shape[0])
warped = cv2.warpPerspective(combined, matP, (img_size[0], img_size[1]))
```
we get the warped image

Note that stretching the rectangle too further away will generate warped image in which lane lines near the top look blurred and not straight enough. Visually this can be metigated by plotting the image in narrow and tall figure size like `figsize=9, 16`

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

To fit a polynomial line to a lane line first I need to collect all the pixels that form a lane line. I accomplished that in `find_lane_line_pixels(image, window_width, window_height, margin)` in `./pipeline_helpers.py` which employed the sliding window approach that involved computing convolution.

After that I computed 2nd order polynomial coefficients by

```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```
where `leftx, lefty, rightx, righty` are just coordinates for left and right lane lines pixels.

As an extra step you can draw the polynomial fits after `plt.imshow()`, fill the polygons wrap up the lane line in green or other color you like and color the lane line pixels in red and blue. The code that does these are in `get_lane_line_bounded_image(warped, left_fitx, right_fitx, ploty, margin)` (for draw the polygons) and `color_unwarped_lane(warped, img_size, left_fitx, right_fitx, ploty, matP_inv)` (for color lane line pixels) in `./pipeline.py`.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code used to do this is in `get_cuvature(leftx, lefty, rightx, righty, ploty)` and `get_car_offset(combined, matP, img_size, bottom_left_fitx, bottom_right_fitx, xm_per_pix)` in `./pipeline.py`. I used the formular provided in the exercise to compute the curvature in meters.

One thing I want to address is how to compute car offset. I chose to compute it in the warped image. So I need the car center and lane center coordinates in the warped image. Lane center is easy once you have the left and right lane line polynomial fits. For car center what I ended up with doing is create a blank image and set the bottom center to 1 and 0 elsewhere. The bottom center is supposed to be the car center in the unwarped image. Then I warp the image and look for nonzero pixel in the bottom. Thus I get my car center in the warped image.

But there could be easier ways. I have been asking questions in the forum and got some answers which I don't fully comprehend at the moment.

[Where is the vehicle in relation to the center of the road?](https://discussions.udacity.com/t/where-is-the-vehicle-in-relation-to-the-center-of-the-road/237424/13)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I obtained image with lane colored in green in `color_unwarped_lane(warped, img_size, left_fitx, right_fitx, ploty, matP_inv)` in `./pipeline.py`, then I overlapped it on top of the undistorted image, in line 464 in `./pipeline.py`.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./detected_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I didn't implement **Look-Ahead Filter** as suggested in the exercise. I wasn't sure about some steps in **Sanity Checks**. I have been shooting questions in the forum and got good answers. But I don't think I'm going to implement it before submission. My thread:

[Sanity check clarification needed](https://discussions.udacity.com/t/sanity-check-clarification-needed/485771)

Also I will argue **Look-Ahead Filter** doesn't save much in terms of computational cost. In sliding window approach we divide the image to many levels by window height. We only do a blind search in the initial scan for the 1st level. The following scans will be based on the result found in the last scan. And computing covolution to get a column sum in some sense is always necessary and can't be avoided by Look-Ahead filter. By ditching Look-Ahead Filter I also saved the computational cost induced by **Sanity Checks**

I did used `Line()` to track the last n detected lane lines to smooth the colored lane area boundaries. The result was positive.

I heard this project can be tackled by Deep Learning. I think it's interesting but collecting training data can be a challenge?
