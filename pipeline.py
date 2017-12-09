import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL
%matplotlib inline

# STEP1: Camera Calibration
# we have many images from the same camera
# we use all of them to calibrate the camera
fn_prefix = 'camera_cal\calibration'
fnames = []
nx = 9
ny = 6
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

# plot distorted and undistorted chessboard images
img_bgr = cv2.imread(fn_prefix + str(5) + '.jpg')
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
undist = cv2.undistort(img, camMat, distCoeffs, None, camMat)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(undist)
ax2.set_title('Undistorted Image', fontsize=30)
plt.subplots_adjust(left=0.04, right=1, top=0.9, bottom=0.)

# save the output image, uncomment to save it to your local disk
# f.savefig('./output_images/calibration_chessboard.jpg')

# plot undistordted road images
img_bgr = cv2.imread('./test_images/test1.jpg')
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
undist = cv2.undistort(img, camMat, distCoeffs, None, camMat)

plt.figure(figsize=(16, 9))
plt.imshow(undist, interpolation='nearest', aspect='auto')

# save the output image, uncomment to save it to your local disk
# plt.savefig('./output_images/calibration_test1.jpg', bbox_inches='tight')

# STEP2: retrieve a grayscale image only contains lane lines
# STEP2-a: first let us play with gradients
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
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

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  mag = np.sqrt(sobelx**2, sobely**2)
  scaled_sobel = np.uint8(255 * mag / np.max(mag))

  binary_output = np.zeros_like(scaled_sobel)
  binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

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

img_bgr = cv2.imread('./test_images/test2.jpg')
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
ksize = 3

gradx = abs_sobel_thresh(img_gray, orient='x', sobel_kernel=ksize, thresh=(20, 160))
grady = abs_sobel_thresh(img_gray, orient='y', sobel_kernel=ksize, thresh=(20, 160))
mag_binary = mag_thresh(img_gray, sobel_kernel=ksize, mag_thresh=(30, 160))
dir_binary = dir_thresh(img_gray, sobel_kernel=15, thresh=(0.7, 1.3))

combined_grad = np.zeros_like(dir_binary)
combined_grad[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

# uncomment the following to plot all 4 threshold gradient binary images
# fig_num = 1
# binaries_grad = [gradx, grady, mag_binary, dir_binary]
#
# for bg in binaries_grad:
#   plt.figure(fig_num, figsize=(16, 9))
#   plt.imshow(bg, interpolation='nearest', aspect='auto', cmap='gray')
#   fig_num += 1

# uncomment the following to plot the combined threshold gradient binary images
# plt.figure(figsize=(16, 9))
# plt.imshow(combined_grad, interpolation='nearest', aspect='auto', cmap='gray')

# STEP@-b: now let us play with color space
thresh = (190, 255)

img_binary = np.zeros_like(img_gray)
img_binary[(img_gray > thresh[0]) & (img_gray <= thresh[1])] = 1

plt.figure(figsize=(16, 9))
plt.imshow(img_binary, interpolation='nearest', aspect='auto', cmap='gray')

img_b = img_bgr[:, :, 0]
img_g = img_bgr[:, :, 1]
img_r = img_bgr[:, :, 2]

binary_b = np.zeros_like(img_b)
binary_g = np.zeros_like(img_g)
binary_r = np.zeros_like(img_r)

binary_b[(img_b > thresh[0]) & (img_b <= thresh[1])] = 1
binary_g[(img_g > thresh[0]) & (img_g <= thresh[1])] = 1
binary_r[(img_r > thresh[0]) & (img_r <= thresh[1])] = 1

# uncomment to plot r, g, b channel masked lane lines binary image
# binaries_color = [binary_b, binary_g, binary_r]
# fig_num = 1

# for bc in binaries_color:
#   plt.figure(fig_num, figsize=(16, 9))
#   plt.imshow(bc, interpolation='nearest', aspect='auto', cmap='gray')
#   fig_num += 1


hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
S = hls[:, :, 2]

thresh = (90, 255)
binary_s = np.zeros_like(S)
binary_s[(S > thresh[0]) & (S <= thresh[1])] = 1

# uncomment to plot s channel masked lane lines binary image
# plt.figure(figsize=(16, 9))
# plt.imshow(binary_s, interpolation='nearest', aspect='auto', cmap='gray')

combined_color = np.zeros_like(img_b)
combined_color[(binary_s == 1) | (binary_r == 1)] = 1

# uncomment to plot color combined lane lines image
# plt.figure(figsize=(16, 9))
# plt.imshow(combined_color, interpolation='nearest', aspect='auto', cmap='gray')

combined = np.zeros_like(img_b)
combined[(combined_grad == 1) | (combined_color == 1)] = 1

# uncomment to plot color and gradients combined lane lines binary image
# plt.figure(figsize=(16, 9))
# plt.imshow(combined, interpolation='nearest', aspect='auto', cmap='gray')

# uncomment to save the final combined image to your local disk
# comb_rescaled = (combined * 255.).astype(np.uint8)
#
# comb = PIL.Image.fromarray(comb_rescaled)
# comb.save("./output_images/binary_lanelines_test2.jpg")
