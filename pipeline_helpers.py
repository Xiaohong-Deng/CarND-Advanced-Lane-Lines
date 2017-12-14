import numpy as np
import cv2

# generate camera calibration parameters
def get_undist_params(fn_prefix='./camera_cal/calibration', nx=9, ny=6):
  """
  Compute parameters needed to undistort the distorted image

  Input
  -----
  fn_prefix : file name prefix which should be the path to the calibration chessboard images

  nx : number of corners in each row of a chessboard image

  ny : number of corners in eahc column of a chessboard image

  Output
  -----
  A 5-element tuple containing objects of different types
  """
  fnames = []
  objpoints = []
  imgpoints = []
  objp = np.zeros((nx*ny, 3), np.float32)
  objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

  for i in range(20):
    fname = fn_prefix + str(i+1) + '.jpg'
    fnames.append(fname)

  for fn in fnames:
    img = cv2.imread(fn)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)

    if ret:
      objpoints.append(objp)
      imgpoints.append(corners)

  ret, camMat, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                              imgpoints,
                                                              img.shape[:-1][::-1],
                                                              None, None)
  return ret, camMat, distCoeffs, rvecs, tvecs

# the following methods are for gradient and color thresholding
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
  """
  Compute binary grayscaled image that captures the lane lines

  Input
  -----
  gray : a gray image

  orient : the axis along which you compute your gradients

  sobel_kernel : the kernel size passed into Sobel()

  thresh : thresholds used to exclude noises

  Output
  -----
  A binary grayscaled image
  """
  if orient == 'x':
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  elif orient == 'y':
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  else:
    raise ValueError('orient must be either x or y')

  abs_sobel = np.absolute(sobel)
  scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
  binary_output = np.zeros_like(scaled_sobel)
  binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

  return binary_output

def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  mag = np.sqrt(sobelx**2, sobely**2)
  scaled_sobel = np.uint8(255 * mag / np.max(mag))

  binary_output = np.zeros_like(scaled_sobel)
  binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

  return binary_output

def dir_thresh(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  abs_sobelx = np.absolute(sobelx)
  abs_sobely = np.absolute(sobely)

  abs_grad_dir = np.arctan2(abs_sobely, abs_sobelx)

  binary_output = np.zeros_like(abs_grad_dir)
  binary_output[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1
  return binary_output

# I implemented the function used to do thresholding gradients but decided not to use it.
# Because a forum mentor said it was terrible with shadows.
# The final output video is the one generated without gradients. Looks fine.
def combine_grad(gray, ksize=3, grad_thresh=(20, 100), magnitude_thresh=(30, 100), direction_thresh=(0.7, 1.57)):
  """
  Compute binary grayscaled image that captures the lane lines

  Input
  -----
  gray : a gray image

  ksize : kernel size to pass into other methods called in this method

  grad_thresh : thresholds passed into abs_sobel_thresh()

  magnitude_thresh : threhsolds passed into mag_thresh()

  direction_thresh : thresholds passed into dir_thresh()

  Output
  -----
  A binary grayscaled image
  """
  gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=grad_thresh)
  grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=grad_thresh)
  mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=magnitude_thresh)
  dir_binary = dir_thresh(gray, sobel_kernel=15, thresh=direction_thresh)

  combined = np.zeros_like(dir_binary)
  combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

  return combined

def combine_color(img_bgr, rgb_thresh=(220, 255), hls_thresh=(90, 255)):
  """
  Compute binary grayscaled image that captures the lane lines

  Input
  -----
  img_bgr : image in BGR form

  rgb_thresh : thresholds for R, G, B channel images to exclude noises

  hls_thresh : thresholds for H, L, S channel images to exclude noises

  Output
  -----
  A binary grayscaled image
  """
  img_r = img_bgr[:, :, 2]
  binary_r = np.zeros_like(img_r)
  binary_r[(img_r > rgb_thresh[0]) & (img_r <= rgb_thresh[1])] = 1

  hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
  S = hls[:, :, 2]
  L = hls[:, :, 1]
  binary_s = np.zeros_like(S)
  binary_l = np.zeros_like(L)
  binary_s[(S > hls_thresh[0]) & (S <= hls_thresh[1])] = 1
  binary_l[(L > hls_thresh[0]) & (L <= hls_thresh[1])] = 1

  combined = np.zeros_like(img_r)
  combined[((binary_s == 1) & (binary_l == 1)) | (binary_r == 1)] = 1

  return combined

# I didn't use window_mask method in my pipeline, skip reading it if you want
# window_mask is used to draw green windows on the lane lines in the image
def window_mask(width, height, img_ref, center,level):
  output = np.zeros_like(img_ref)
  output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
          max(0,int(center-width/2)):min(int(center+width/2),
                                         img_ref.shape[1])] = 1
  return output

def find_lane_line_pixels(image, window_width, window_height, margin):
  """
  Computes all the coordinates for the pixels that constitute the lane lines
  Use convolution to find window centroid. Then the lane line pixels

  Inputs
  -----
  image : presumably it is a binary grayscaled image, with elements of only 0 or 1

  window_width : covolution window width of your choice

  window_height : covolution widnow height of your choice

  margin : the horizontal offset from the window centroids we use to draw a bounding box
           at each level to find lane line pixels. Dont confuse it with covolution window width

  Outputs
  -----
  a 4-element tuple containing x coordinates and y coordinates for left and right lane lines
  """
  window_centroids = [] # Store the (left,right) window centroid positions per level
  window = np.ones(window_width) # Create our window template that we will use for convolutions

  # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
  # and then np.convolve the vertical image slice with the window template

  # Sum quarter bottom of image to get slice, could use a different ratio
  # image is a grayscale, looking at lower left quarter
  l_sum = np.sum(image[int(image.shape[0]/2):,:int(image.shape[1]/2)], axis=0)
  l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
  # the lower right quarter
  r_sum = np.sum(image[int(image.shape[0]/2):,int(image.shape[1]/2):], axis=0)
  # the convolution starts from index 0, so we shift it by half of the width
  r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)

  # note l_center and r_center are x coordinates in the image

  # Add what we found for the first layer
  # this is our first window
  window_centroids.append((l_center, r_center))

  # still we need to collect all the nonzero pixels in this window
  # so later we can fit a polynomial
  # here the window width used to collect pixels is 2 * margin
  nonzero = image.nonzero()
  nonzeroy = nonzero[0]
  nonzerox = nonzero[1]

  left_lane_inds = []
  right_lane_inds = []

  win_y_low = image.shape[0] - window_height
  win_y_high = image.shape[0]

  win_x_l_low = l_center - margin
  win_x_l_high = l_center + margin
  win_x_r_low = r_center - margin
  win_x_r_high = r_center + margin

  good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= win_x_l_low) & (nonzerox < win_x_l_high)).nonzero()[0]
  good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_x_r_low) & (nonzerox < win_x_r_high)).nonzero()[0]

  left_lane_inds.append(good_left_inds)
  right_lane_inds.append(good_right_inds)

  # Go through each layer looking for max pixel locations
  for level in range(1,(int)(image.shape[0]/window_height)):
    # convolve the window into the vertical slice of the image
    # in the loop we go through the entire width
    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
    conv_signal = np.convolve(window, image_layer)
    # Find the best left centroid by using past left center as a reference
    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
    offset = window_width/2
    # to avoid negative index, use max()
    # it is the index in conv_signal
    l_min_index = int(max(l_center+offset-margin,0))
    # to avoid index larger than width, use min()
    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
    # get the index in original image
    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
    # Find the best right centroid by using past right center as a reference
    r_min_index = int(max(r_center+offset-margin,0))
    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
    # Add what we found for that layer
    window_centroids.append((l_center,r_center))

    win_y_low = image.shape[0] - (level + 1) * window_height
    win_y_high = image.shape[0] - level * window_height

    win_x_l_low = l_center - margin
    win_x_l_high = l_center + margin
    win_x_r_low = r_center - margin
    win_x_r_high = r_center + margin

    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_x_l_low) & (nonzerox < win_x_l_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_x_r_low) & (nonzerox < win_x_r_high)).nonzero()[0]

    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)
  # centroids we get is the x coordinates for n windows
  # Extract left and right line pixel positions
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds]
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]

  # Fit a second order polynomial to each
  left_fit = np.polyfit(lefty, leftx, 2)
  right_fit = np.polyfit(righty, rightx, 2)

  # Generate x and y values for plotting
  ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

  return leftx, lefty, rightx, righty
