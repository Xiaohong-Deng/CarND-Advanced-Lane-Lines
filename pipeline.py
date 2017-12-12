import pickle
import matplotlib.pyplot as plt
import PIL
import pipeline_helpers as ph
import numpy as np
import cv2
from importlib import reload
%matplotlib inline

# STEP1: Camera Calibration
# we have many chessboard images from the same camera
# we use all of them to calibrate the camera

# load the undistort parameters for future use, or run the method to generate
# and save them
def get_undist_params():
  """
  Compute parameters needed to undistort the distorted image

  Output
  -----
  A 5-element tuple containing objects of different types
  """
  try:
    with open('undist_params.p', mode='rb') as f:
      undist_params = pickle.load(f)
      ret, camMat, distCoeffs, rvecs, tvecs = undist_params['ret'], \
                                              undist_params['camMat'], \
                                              undist_params['distCoeffs'], \
                                              undist_params['rvecs'], \
                                              undist_params['tvecs']
  except FileNotFoundError:
    undist_params = {}
    ret, camMat, distCoeffs, rvecs, tvecs = ph.get_undist_params()
    undist_params['ret'], undist_params['camMat'], undist_params['distCoeffs'], \
    undist_params['rvecs'], undist_params['tvecs'] = ret, camMat, distCoeffs, rvecs, tvecs

    with open('undist_params.p', mode='wb') as f:
      pickle.dump(undist_params, f)

  return ret, camMat, distCoeffs, rvecs, tvecs

# load and preprocess the image
def get_all_imgs(img_bgr):
  img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

  return img, img_gray

def get_perspective_mat(camMat, distCoeffs):
  try:
    with open('perspective_params.p', mode='rb') as f:
      perspective_params = pickle.load(f)
      matP, matP_inv = perspective_params['matP'], perspective_params['matP_inv']
  except FileNotFoundError:
    straight_line_bgr = cv2.imread('./test_images/straight_lines2.jpg')
    undist_straight_line = cv2.undistort(straight_line_bgr, camMat, distCoeffs, None, camMat)

    # uncomment the following line to save undistorted straight line image to your local disk
    # after storing the undistorted image we can eyeball the 4 rectangle corners in the image
    # cv2.imwrite('./output_images/undist_straight_line.jpg', undist_straight_line)
    # coordinates are in (width, hieght) or (x, y)
    # following are the 4 corners of a rectangle in the unwarped image by eyeballing
    # with some software help

    upper_left = [578, 462]
    upper_right = [704, 462]
    lower_left = [212, 719]
    lower_right = [1093, 719]

    offset_x = 300
    offset_y = 0

    src = np.float32([upper_left, upper_right, lower_right, lower_left])
    dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x - 1, offset_y],
                      [img_size[0] - offset_x - 1, img_size[1] - offset_y - 1],
                      [offset_x, img_size[1] - offset_y - 1]])

    matP = cv2.getPerspectiveTransform(src, dst)
    matP_inv = cv2.getPerspectiveTransform(dst, src)

    perspective_params = {}
    perspective_params['matP'], perspective_params['matP_inv'] = matP, matP_inv
    with open('perspective_params.p', mode='wb') as f:
      pickle.dump(perspective_params, f)

  return matP, matP_inv

def get_bin_lane_line_img(img_gray, img_bgr, ksize=3):
  combined_grad = ph.combine_grad(img_gray, ksize=ksize)
  # STEP2-b: get the best lane lines captured image we can with color space
  combined_color, bin_s, bin_l = ph.combine_color(img_bgr)
  combined = np.zeros_like(combined_color)
  combined[(combined_grad == 1) | (combined_color == 1)] = 1

  return combined

def color_warped_lane_lines(warped, leftx, lefty, rightx, righty):
  warped[lefty, leftx] = [255, 0, 0] # left line in red
  warped[righty, rightx] = [0, 0, 255] # right line in blue

  return warped

def get_lane_line_bounded_image(warped, left_fitx, right_fitx, ploty, margin):
  # STEP3-b: now we find the lane lines from the images using convolution
  # draw windows around them, fit a polynomial and draw it on the image

  window_img = np.zeros_like(warped)
  # Generate a polygon to illustrate the search window area
  # And recast the x and y points into usable format for cv2.fillPoly()
  # vstack is vertical stack, more rows
  # transpose gives you pairs of pixel coordinates, (x, y), it gives you left lane line
  # left boundary from top to bottom
  left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
  # it gives you left lane line right boundary from bottom to top, by flipping upside down
  left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                ploty])))])

  # (left_x, y, right_x, y), y is the same, so it gives you the smooth boundaries of lane lines
  # left_line_window1 and left_line_window2 are of the shape(1, 720, 2)
  # horizontal stack is stacking the 2nd dimension, (1, 1440, 2)
  left_line_pts = np.hstack((left_line_window1, left_line_window2))
  right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
  right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                ploty])))])
  right_line_pts = np.hstack((right_line_window1, right_line_window2))

  # Draw the lane onto the warped blank image
  cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
  cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

  return window_img

def get_cuvature(leftx, lefty, rightx, righty, ploty):
  ym_per_pix = 30/720 # meters per pixel in y dimension
  xm_per_pix = 3.7/700 # meters per pixel in x dimension

  y_eval = np.max(ploty)

  # Fit new polynomials to x,y in world space
  left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
  right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
  # Calculate the new radii of curvature
  left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
  right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

  return left_curverad, right_curverad

def get_car_offset(combined, matP, img_size, bottom_left_fitx, bottom_right_fitx, xm_per_pix):
  # also I want to know where the car center is in the warped image
  # car center is assumed to be horizontally in the center of the unwarped image
  car_center = np.zeros_like(combined)
  car_center[car_center.shape[0] - 1, car_center.shape[1] // 2 - 1] = 1
  car_center_warp = cv2.warpPerspective(car_center, matP, (img_size[0], img_size[1]))
  car_centerx = np.argmax(car_center_warp[car_center_warp.shape[0] - 1, :])
  lane_centerx = ((bottom_right_fitx + bottom_left_fitx) // 2)

  car_offset_meters = (car_centerx - lane_centerx) * xm_per_pix

  return car_offset_meters

def color_unwarped_lane(warped, img_size, left_fitx, right_fitx, ploty):
  # Let us try to draw the lane in green and warp it back to the original perspective
  # Create an image to draw the lines on
  warp_zero = np.zeros_like(warped).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

  # Recast the x and y points into usable format for cv2.fillPoly()
  pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
  pts = np.hstack((pts_left, pts_right))

  # Draw the lane onto the warped blank image
  cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

  # Warp the blank back to original image space using inverse perspective matrix (Minv)

  # Combine the result with the original image
  newwarp = cv2.warpPerspective(color_warp, matP_inv, (img_size[0], img_size[1]))

  return newwarp

def paste_curvature_and_offset(image, left_curverad, right_curverad, offset):
  font = cv2.FONT_HERSHEY_SIMPLEX
  image = cv2.putText(image,"left line curvature: " + str(left_curverad) + " meters", (20,40), font, 1, (255,255,255), 2, cv2.LINE_AA)
  image = cv2.putText(image,"right line curvature: " + str(right_curverad) + " meters", (20,80), font, 1, (255,255,255), 2, cv2.LINE_AA)
  image = cv2.putText(image,"car offset: " + str(car_offset_meters) + " meters" , (20,120), font, 1, (255,255,255), 2, cv2.LINE_AA)

  return image

def process_image(img_bgr):
  ret, camMat, distCoeffs, rvecs, tvecs = get_undist_params()
  matP, matP_inv = get_perspective_mat(camMat, distCoeffs)
  img, img_gray = get_all_imgs(img_bgr)
  undist = cv2.undistort(img, camMat, distCoeffs, None, camMat)
  combined = get_bin_lane_line_img(img_gray, img_bgr)

  # STEP3: let us warp the image to bird's eyes view perspective
  # STEP3-a: either we load the matrices that can be used to transform images to bird eye's
  # view perspective or we generate such matrices from a straight line image
  img_size = (undist.shape[1], undist.shape[0])
  warped = cv2.warpPerspective(combined, matP, (img_size[0], img_size[1]))

  # window settings
  # this is the window_width used to do convolution
  window_width = 50
  window_height = 180 # Break image into 9 vertical layers since image height is 720
  margin = 100 # How much to slide left and right for searching

  # find the lane lines centers in the bird eye's perspective
  leftx, lefty, rightx, righty = ph.find_lane_line_pixels(warped, window_width, window_height, margin)

  # Fit a second order polynomial to each
  left_fit = np.polyfit(lefty, leftx, 2)
  right_fit = np.polyfit(righty, rightx, 2)

  # Generate x and y values for plotting
  ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

  out_img = np.dstack((warped, warped, warped))*255
  out_img = color_warped_lane_lines(out_img, leftx, lefty, rightx, righty)
  lane_line_bounded = get_lane_line_bounded_image(out_img, left_fitx, right_fitx, ploty, margin)

  colored_lane_line_bounded = cv2.addWeighted(out_img, 1, lane_line_bounded, 0.3, 0)

  left_curverad, right_curverad = get_cuvature(leftx, lefty, rightx, righty, ploty)
  car_offset = get_car_offset(combined, matP, img_size, left_fitx[-1], right_fitx[-1], xm_per_pix)

  colored_lane = color_unwarped_lane(warped, img_size, left_fitx, right_fitx, ploty)
  colored_lane_img = cv2.addWeighted(undist, 1, colored_lane, 0.3, 0)

  return colored_lane_line_bounded, colored_lane_img
# STEP2: retrieve a grayscale image only contains lane lines
# STEP2-a: get the best lane lines captured image we can with gradients

img_bgr = cv2.imread('./test_images/test4.jpg')
warped_colored_lane_line, colored_lane_img = process_image(img_bgr)

plt.figure(figsize=(16, 9))
plt.imshow(warped_colored_lane_line)
plt.figure(figsize=(16, 9))
plt.imshow(colored_lane_img)
# plt.plot(left_fitx, ploty, color='yellow') # center line for left line in yellow
# plt.plot(right_fitx, ploty, color='yellow') # center line for right line in yellow
# plt.xlim(0, 1280)
# plt.ylim(720, 0)
