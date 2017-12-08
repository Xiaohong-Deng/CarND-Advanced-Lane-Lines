import numpy as np
import cv2
import matplotlib.pyplot as plt
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

img = cv2.imread(fn_prefix + str(5) + '.jpg')
undist = cv2.undistort(img, camMat, distCoeffs, None, camMat)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undist)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
