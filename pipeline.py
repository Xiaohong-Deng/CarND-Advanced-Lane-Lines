import pickle
import matplotlib.pyplot as plt
import PIL
import pipeline_helpers as ph
import numpy as np
import cv2
import imageio
from moviepy.editor import VideoFileClip

# Define a class to receive the characteristics of each line detection
class Line():
  def __init__(self):
    # was the line detected in the last iteration?
    self.detected = False
    # x values of the last n fits of the line
    self.recent_xfitted = []
    #average x values of the fitted line over the last n iterations
    self.bestx = None
    #polynomial coefficients averaged over the last n iterations
    self.best_fit = None
    #polynomial coefficients for the most recent fit
    self.current_fit = [np.array([False])]
    #radius of curvature of the line in some units
    self.radius_of_curvature = None
    #distance in meters of vehicle center from the line
    self.line_base_pos = None
    #difference in fit coefficients between last and new fits
    self.diffs = np.array([0,0,0], dtype='float')
    #x values for detected line pixels
    self.allx = None
    #y values for detected line pixels
    self.ally = None

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
def get_all_imgs(img, is_bgr):
  """
  Return the same images in 3 different color form

  Input
  -----
  img : an image that is either in RGB or BGR form

  is_bgr : Boolean value indicating if 'img' is in BGR form

  Output
  -----
  img : image in RGB form

  img_bgr : image in BGR form

  img_gray : image in grayscale form
  """
  if is_bgr:
    img_bgr = img
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  else:
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

  return img, img_bgr, img_gray

def get_perspective_mat(camMat, distCoeffs):
  """
  Return matrices used to transform images between different perspectives

  Input
  -----
  camMat : calibration matrix use to undistort images

  distCoeffs : distortion coefficients used to undistort images

  Output
  -----
  matP : matrix used to transform images from original to bird eye's view

  matP_inv : matrix used to transform images from bird eye's view to original
  """
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

def get_bin_lane_line_img(img_gray, img_bgr):
  """
  Get the best lane lines captured image we can with color space

  Input
  -----
  img_gray : an image in grayscale form

  img_bgr : an image in BGR form

  Output
  -----
  combined_color : lane lines captured binary grayscaled image by merging images from different color channels
  """
  combined_color = ph.combine_color(img_bgr)

  return combined_color

def color_warped_lane_lines(warped, leftx, lefty, rightx, righty):
  """
  For the warped image, set pixels that form the left line in red, pixels that
  form the right line in blue
  """
  warped[lefty, leftx] = [255, 0, 0] # left line in red
  warped[righty, rightx] = [0, 0, 255] # right line in blue

  return warped

def get_lane_line_bounded_image(warped, left_fitx, right_fitx, ploty, margin):
  """
  Return a warped image that within a margin along the lane lines colored in green

  Input
  -----
  warped : original warped image

  left_fitx : x coordinates for the left fitted polynomial line

  right_fitx : x coordinates for the right fitted polynomial line

  ploty : y coordinates for left and right polynomial lines

  margin : offset from the polynomial lines that determines the width of the colored area
  """
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
  """
  Return radiuses of curvature for left and right lane lines

  Input
  -----
  leftx : x coordinates for pixels that form the left lane line

  lefty : y coordinates for pixels that form the left lane line

  rightx : x coordinates for pixels that form the right lane line

  righty : y coordinates for pixels that form the right lane line

  ploty : y coordinates for left and right polynomial lines

  Output
  -----
  left_curverad : Radius of curvature measured in meters for left lane line.
                  It is measured at the bottom of the image
  right_curverad : Radius of curvature measured in meters for right lane line.
                  It is measured at the bottom of the image
  """
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
  """
  Return the measurement of how much car center is off the lane center, in meters

  Input
  -----
  combined : the undistorted, binary grayscaled image in original perspective

  matP : matrix used to transform images from original to bird eye's view

  img_size : a tuple containing (image_width, image_height)

  bottom_left_fitx : x coordinate for the left line polynomial fit when y = 719

  bottom_right_fitx : x coordinate for the right line polynomial fit when y = 719

  xm_per_pix : measurement on how many meters changed by increasing or decreasing a pixel horizontally
               in the image

  Output
  -----
  car_offset_meters : measurement in meters on how much car center is off the lane center
  """
  # also I want to know where the car center is in the warped image
  # car center is assumed to be horizontally in the center of the unwarped image
  # I mark the center at the bottom of the unwarped image, warp the image, find
  # the marked point in the warped image then I get the car center in the warped image
  car_center = np.zeros_like(combined)
  car_center[car_center.shape[0] - 1, car_center.shape[1] // 2 - 1] = 1
  car_center_warp = cv2.warpPerspective(car_center, matP, (img_size[0], img_size[1]))
  car_centerx = np.argmax(car_center_warp[car_center_warp.shape[0] - 1, :])
  lane_centerx = ((bottom_right_fitx + bottom_left_fitx) // 2)

  car_offset_meters = (car_centerx - lane_centerx) * xm_per_pix

  return car_offset_meters

def color_unwarped_lane(warped, img_size, left_fitx, right_fitx, ploty, matP_inv):
  """
  Return an image in the unwarped perspective with lane colored in green

  Input
  -----
  warped : warped image in bird eye's perspective

  img_size : a tuple containing (image_width, image_height)

  left_fitx : x coordinates for the left fitted polynomial line

  right_fitx : x coordinates for the right fitted polynomial line

  ploty : y coordinates for left and right polynomial lines

  matP_inv : matrix used to transform images from bird eye's view to original
  """
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
  newwarp = cv2.warpPerspective(color_warp, matP_inv, (img_size[0], img_size[1]))

  return newwarp

def paste_curvature_and_offset(image, curverad, offset):
  """
  Return image with curvature and car offset information embedded

  Input
  -----
  image : image to be modified

  curverad : Radius of curvature in meters

  offset : measurement in meters on how much car center is off the lane center
  """
  font = cv2.FONT_HERSHEY_SIMPLEX
  image = cv2.putText(image,"lane curvature: " + str(curverad) + " meters", (20,40), font, 1, (255,255,255), 2, cv2.LINE_AA)
  image = cv2.putText(image,"car offset: " + str(offset) + " meters" , (20,120), font, 1, (255,255,255), 2, cv2.LINE_AA)

  return image


def update_line(line, fitx, fit):
  """
  Update Line() instance variable

  Input
  -----
  line : the Line() instance to be updated

  fitx : x coordinates for the fitted polynomial line

  fit : 2nd order polynomial coefficients
  """
  line.detected = True
  num_tracked_lines = len(line.recent_xfitted)

  if num_tracked_lines == 10:
    line.recent_xfitted.pop(0)
  line.recent_xfitted.append(fitx)

  line.bestx = np.mean(line.recent_xfitted, axis=0)

  if line.best_fit is None:
    line.best_fit = fit
  else:
    if num_tracked_lines == 10:
      line.best_fit = (line.best_fit * num_tracked_lines + fit) / num_tracked_lines
    else:
      line.best_fit = (line.best_fit * num_tracked_lines + fit) / (num_tracked_lines + 1)

  line.diffs = fit - line.current_fit
  line.current_fit = fit

def process_frames(is_bgr=True, left_line=None, right_line=None):
  """
  Return a pipeline function that can process a single image

  Input
  -----
  is_bgr : parameter that decide if the returned function takes BGR images or RGB images

  left_line : the Line() instance used to keep track of the information of the
              detected left lines in the last n frames
  right_line : the Line() instance used to keep track of the information of the
              detected right lines in the last n frames
  """
  if left_line is None:
    left_line = Line()
  if right_line is None:
    right_line = Line()

  def process_image(img):
    # STEP1: Camera Calibration
    # we have many chessboard images from the same camera
    # we use all of them to calibrate the camera
    ret, camMat, distCoeffs, rvecs, tvecs = get_undist_params()
    matP, matP_inv = get_perspective_mat(camMat, distCoeffs)
    img, img_bgr, img_gray = get_all_imgs(img, is_bgr)
    undist = cv2.undistort(img, camMat, distCoeffs, None, camMat)
    # STEP2: retrieve a grayscale image only contains lane lines
    # STEP2-a: get the best lane lines captured image we can with gradients
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
    xm_per_pix = 3.7/700

    # find the lane lines centers in the bird eye's perspective
    leftx, lefty, rightx, righty = ph.find_lane_line_pixels(warped, window_width, window_height, margin)

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_curverad, right_curverad = get_cuvature(leftx, lefty, rightx, righty, ploty)
    curverad = (left_curverad + right_curverad) / 2

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    update_line(left_line, left_fitx, left_fit)
    update_line(right_line, right_fitx, right_fit)

    # out_img = np.dstack((warped, warped, warped))*255
    # out_img = color_warped_lane_lines(out_img, leftx, lefty, rightx, righty)
    # lane_line_bounded = get_lane_line_bounded_image(out_img, left_fitx, right_fitx, ploty, margin)

    # colored_lane_line_bounded = cv2.addWeighted(out_img, 1, lane_line_bounded, 0.3, 0)

    car_offset = get_car_offset(combined, matP, img_size, left_fitx[-1], right_fitx[-1], xm_per_pix)

    colored_lane = color_unwarped_lane(warped, img_size, left_line.bestx, right_line.bestx, ploty, matP_inv)
    colored_lane_img = cv2.addWeighted(undist, 1, colored_lane, 0.3, 0)
    colored_lane_img = paste_curvature_and_offset(colored_lane_img, curverad, car_offset)

    return colored_lane_img

  return process_image

if name == '__main__':
  clip1 = VideoFileClip("./project_video.mp4")

  left_line = Line()
  right_line = Line()
  result = clip1.fl_image(process_frames(is_bgr=False, left_line=left_line, right_line=right_line))
  result.write_videofile('./detected_project_video.mp4', audio=False)
