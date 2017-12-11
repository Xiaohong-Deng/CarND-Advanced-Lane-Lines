import numpy as np
import cv2

# generate camera calibration parameters
def get_undist_params():
  fn_prefix = './camera_cal/calibration'
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
  return ret, camMat, distCoeffs, rvecs, tvecs

# the following methods are for gradient and color thresholding
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

def combine_grad(gray, ksize=3, grad_thresh=(20, 100), magnitude_thresh=(30, 100), direction_thresh=(0.7, 1.57)):
  gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=grad_thresh)
  grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=grad_thresh)
  mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=magnitude_thresh)
  dir_binary = dir_thresh(gray, sobel_kernel=15, thresh=direction_thresh)

  combined = np.zeros_like(dir_binary)
  combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

  return combined

def combine_color(img_bgr, rgb_thresh=(220, 255), hls_thresh=(90, 255)):
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

  return combined, binary_s, binary_l
