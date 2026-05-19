import cv2 as cv
import numpy as np
import argparse

# Paths to the binary images
image_data_2832 = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/4_coverage_map/traditional_compiled_binary_map_50.png"
image_data_0036 = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_0036/4_coverage_map/traditional_compiled_binary_map_50.png"

# Argument parser for input images
parser = argparse.ArgumentParser(description='Code for Feature Matching with ORB.')
parser.add_argument('--input1', help='Path to input image 1.', default=image_data_2832)
parser.add_argument('--input2', help='Path to input image 2.', default=image_data_0036)
args = parser.parse_args()

# Load the images
img_object = cv.imread(cv.samples.findFile(args.input1), cv.IMREAD_GRAYSCALE)
img_scene = cv.imread(cv.samples.findFile(args.input2), cv.IMREAD_GRAYSCALE)
if img_object is None or img_scene is None:
    print('Could not open or find the images!')
    exit(0)

# -- Step 1: Detect the keypoints using ORB Detector and compute the descriptors
orb = cv.ORB_create()
keypoints_obj, descriptors_obj = orb.detectAndCompute(img_object, None)
keypoints_scene, descriptors_scene = orb.detectAndCompute(img_scene, None)

# -- Step 2: Matching descriptor vectors with a BFMatcher
# ORB uses binary descriptors, so NORM_HAMMING is used
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors_obj, descriptors_scene)

# -- Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# -- Draw matches
img_matches = np.empty(
    (max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1] + img_scene.shape[1], 3), dtype=np.uint8
)
cv.drawMatches(
    img_object,
    keypoints_obj,
    img_scene,
    keypoints_scene,
    matches[:50],  # Draw the top 50 matches
    img_matches,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# -- Localize the object
obj = np.empty((len(matches), 2), dtype=np.float32)
scene = np.empty((len(matches), 2), dtype=np.float32)
for i in range(len(matches)):
    # -- Get the keypoints from the matches
    obj[i, 0] = keypoints_obj[matches[i].queryIdx].pt[0]
    obj[i, 1] = keypoints_obj[matches[i].queryIdx].pt[1]
    scene[i, 0] = keypoints_scene[matches[i].trainIdx].pt[0]
    scene[i, 1] = keypoints_scene[matches[i].trainIdx].pt[1]

# -- Estimate homography matrix
H, _ = cv.findHomography(obj, scene, cv.RANSAC)

# -- Get the corners from the image_1 (the object to be "detected")
obj_corners = np.empty((4, 1, 2), dtype=np.float32)
obj_corners[0, 0, 0] = 0
obj_corners[0, 0, 1] = 0
obj_corners[1, 0, 0] = img_object.shape[1]
obj_corners[1, 0, 1] = 0
obj_corners[2, 0, 0] = img_object.shape[1]
obj_corners[2, 0, 1] = img_object.shape[0]
obj_corners[3, 0, 0] = 0
obj_corners[3, 0, 1] = img_object.shape[0]

scene_corners = cv.perspectiveTransform(obj_corners, H)

# -- Draw lines between the corners (the mapped object in the scene - image_2)
cv.line(
    img_matches,
    (int(scene_corners[0, 0, 0] + img_object.shape[1]), int(scene_corners[0, 0, 1])),
    (int(scene_corners[1, 0, 0] + img_object.shape[1]), int(scene_corners[1, 0, 1])),
    (0, 255, 0),
    4,
)
cv.line(
    img_matches,
    (int(scene_corners[1, 0, 0] + img_object.shape[1]), int(scene_corners[1, 0, 1])),
    (int(scene_corners[2, 0, 0] + img_object.shape[1]), int(scene_corners[2, 0, 1])),
    (0, 255, 0),
    4,
)
cv.line(
    img_matches,
    (int(scene_corners[2, 0, 0] + img_object.shape[1]), int(scene_corners[2, 0, 1])),
    (int(scene_corners[3, 0, 0] + img_object.shape[1]), int(scene_corners[3, 0, 1])),
    (0, 255, 0),
    4,
)
cv.line(
    img_matches,
    (int(scene_corners[3, 0, 0] + img_object.shape[1]), int(scene_corners[3, 0, 1])),
    (int(scene_corners[0, 0, 0] + img_object.shape[1]), int(scene_corners[0, 0, 1])),
    (0, 255, 0),
    4,
)

# -- Show detected matches
cv.imshow('Good Matches & Object detection', img_matches)

cv.waitKey()
