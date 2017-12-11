import pickle
import matplotlib.pyplot as plt
import PIL
import pipeline_helpers as ph
from importlib import reload
%matplotlib inline

# STEP1: Camera Calibration
# we have many chessboard images from the same camera
# we use all of them to calibrate the camera

# load the undistort parameters for future use, or run the method to generate
# and save them
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

# save the distorted and undistorted chessboard image, uncomment to save it to your local disk
# f.savefig('./output_images/calibration_chessboard.jpg')

# plot undistordted road images
img_bgr = cv2.imread('./test_images/test4.jpg')
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
undist = cv2.undistort(img, camMat, distCoeffs, None, camMat)

# uncomment to plot undistorted road image
# plt.figure(figsize=(16, 9))
# plt.imshow(undist, interpolation='nearest', aspect='auto')

# save the output image, uncomment to save it to your local disk
# plt.savefig('./output_images/calibration_test1.jpg', bbox_inches='tight')

# STEP2: retrieve a grayscale image only contains lane lines
# STEP2-a: first let us play with gradients

# test our thresholding gradients methods
ksize = 3

combined_grad = ph.combine_grad(img_gray, ksize=ksize)
# STEP2-b: now let us play with color space

reload(ph)
combined_color, bin_s, bin_l = ph.combine_color(img_bgr)
combined = np.zeros_like(combined_color)
combined[(combined_grad == 1) | (combined_color == 1)] = 1
# uncomment to plot color and gradients combined lane lines binary image
plt.figure(figsize=(16, 9))
plt.imshow(combined_grad, interpolation='nearest', aspect='auto', cmap='gray')

# uncomment to save the final combined image to your local disk
# comb_rescaled = (combined * 255.).astype(np.uint8)
#
# comb = PIL.Image.fromarray(comb_rescaled)
# comb.save("./output_images/binary_lanelines_test2.jpg")

# STEP3: let us warp the image to bird's eyes view perspective
# STEP3-a: either we load the matrices that can be used to transform images to bird eye's
# view perspective or we generate such matrices from a straight line image
img_size = (undist.shape[1], undist.shape[0])

try:
  with open('perspective_params.p', mode='rb') as f:
    perspective_params = pickle.load(f)
    matP, matP_inv = perspective_params['matP'], perspective_params['matP_inv']
except FileNotFoundError:
  straight_line_bgr = cv2.imread('./test_images/straight_lines1.jpg')
  undist_straight_line = cv2.undistort(straight_line_bgr, camMat, distCoeffs, None, camMat)

  # uncomment the following line to save undistorted straight line image to your local disk
  # after storing the undistorted image we can eyeball the 4 rectangle corners in the image
  # cv2.imwrite('./output_images/undist_straight_line.jpg', undist_straight_line)
  # coordinates are in (width, hieght) or (x, y)
  # following are the 4 corners of a rectangle in the unwarped image by eyeballing
  # with some software help

  upper_left = [585, 460]
  upper_right = [701, 460]
  lower_left = [211, 719]
  lower_right = [1093, 719]

  offset_x = 300
  offset_y = 0

  src = np.float32([upper_left, upper_right, lower_right, lower_left])
  dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x, offset_y],
                    [img_size[0] - offset_x, img_size[1] - offset_y],
                    [offset_x, img_size[1] - offset_y]])

  matP = cv2.getPerspectiveTransform(src, dst)
  matP_inv = cv2.getPerspectiveTransform(dst, src)

  perspective_params = {}
  perspective_params['matP'], perspective_params['matP_inv'] = matP, matP_inv
  with open('perspective_params.p', mode='wb') as f:
    pickle.dump(perspective_params, f)

# test perspective transformation
warped = cv2.warpPerspective(combined, matP, (img_size[0], img_size[1]))

warped_rescaled = (warped * 255.).astype(np.uint8)
warped_rescaled = PIL.Image.fromarray(warped_rescaled)
warped_rescaled.save("./output_images/binary_warped_test1.jpg")

plt.figure(figsize=(16, 9))
plt.imshow(warped, cmap='gray')

# STEP3-b: now we find the lane lines from the images using convolution
# draw windows around them, fit a polynomial and draw it on the image

# window settings
# this is the window_width used to do convolution
window_width = 50
window_height = 180 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

# window_mask is used to draw green windows in the image
def window_mask(width, height, img_ref, center,level):
  output = np.zeros_like(img_ref)
  output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
          max(0,int(center-width/2)):min(int(center+width/2),
                                         img_ref.shape[1])] = 1
  return output

def find_lane_line_pixels(image, window_width, window_height, margin):

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
# find the lane lines centers in the bird eye's perspective
leftx, lefty, rightx, righty = find_lane_line_pixels(warped, window_width, window_height, margin)

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Generate x and y values for plotting
ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img = np.dstack((warped, warped, warped))*255
window_img = np.zeros_like(out_img)
out_img[lefty, leftx] = [255, 0, 0] # left line in red
out_img[righty, rightx] = [0, 0, 255] # right line in blue
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
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

plt.figure(figsize=(16, 9))
plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow') # center line for left line in yellow
plt.plot(right_fitx, ploty, color='yellow') # center line for right line in yellow
plt.xlim(0, 1280)
plt.ylim(720, 0)

plt.savefig('./output_images/line_pixels_poly_fit_test4.jpg', bbox_inches='tight')

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

y_eval = np.max(ploty)

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
print(left_curverad, 'm', right_curverad, 'm')
# Example values: 632.1 m    626.2 m
