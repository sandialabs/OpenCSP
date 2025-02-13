from math import sqrt as sqrt
from math import ceil as ceil
from math import floor as floor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import logging as log
import multiprocessing as mp
import numpy as np
from cv2 import cv2 as cv
import os
import subprocess
import matplotlib.pyplot as plt

# ## CAMERA MODEL - SONY [frame]
# # FX              = 3434.8 #5403.4
# # FY              = 3403.7 #5318.8
# # CX              = 3516.1 #3852.5
# # CY              = 2000.2   #3785.5

# # K1              = -0.08357355   #-0.280936
# # K2              = 0.04066407    #0.316160
# # K3              = -0.00674798   #-0.119680     #-0.00674798
# # P1              = 0.00936359    #0.003528
# # P2              = -0.00346739   #-0.017712


# ## CAMERA MODEL - SONY
# FX              = 4801.9 #5403.4
# FY              = 4801.9 #5318.8
# CX              = 3840   #3852.5
# CY              = 2160   #3785.5

# K1              = -0.280936
# K2              = 0.316160
# K3              = -0.119680
# P1              = 0.003528
# P2              = -0.017712

# DistCoefs       = np.array([[K1, K2, P1, P2, K3]])
# CameraMatrix    = np.array([[FX,    0,      CX],
#                             [0,     FY,     CY],
#                             [0,     0,      1]]).reshape(3,3)

# CAMERA MODEL - MAVIC  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
FX = 2868.1  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
FY = 2875.9  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
# distortion coefficients  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
K1 = -0.024778  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
K2 = 0.012383  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
# tangential  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
P1 = -0.00032978  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
P2 = -0.0001401  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
## Optical Centers  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
CX = 3840 / 2  # W / 2  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
CY = 2160 / 2  # H / 2   # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
# ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
# ?? SCAFFOLDING RCB -- "DistCoefs"  MISSPELLED -- SHOULD BE "DistCoeffs", OR EVEN BETTER, "DISTORTION_COEFFICIENTS"
DistCoefs = np.array([[K1, K2, P1, P2]])  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
ZeroDistCoefs = np.array([[1.0e-9, 0.0, 0.0, 0.0]])  # ?? SCAFFOLDING RCB -- SET FLOAT TYPE PROPERLY
CameraMatrix = np.array(
    [
        [FX, 0, CX],  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
        [0, FY, CY],  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])
        [0, 0, 1],
    ]
).reshape(
    3, 3
)  # ?? SCAFFOLDING RCB -- USED TO BE SONY ABOVE (NOT [FRAME])


## Boundary Pixels
IGNORE_MARGIN = 3  # 4 #150  # ?? SCAFFOLDING RCB -- CHANGED FROM ORIGINAL VALUE OF 150.  THEORY IS THAT SWITCH TO SONY DROVE A DIFFERENT NUMBER.
REQUIRED_SKY_WIDTH = 80  # 100 #150  # ?? SCAFFOLDING RCB -- CHANGED FROM ORIGINAL VALUE OF 150.  THEORY IS THAT SWITCH TO SONY DROVE A DIFFERENT NUMBER.
REQUIRED_NON_SKY_WIDTH = 0
LEFT_BOUNDARY_COLOR = [255, 0, 0]
RIGHT_BOUNDARY_COLOR = [0, 0, 255]
BOTTOM_BOUNDARY_COLOR = [0, 255, 0]
TOP_BOUNDARY_COLOR = [0, 255, 255]  # ?? SCAFFOLDING RCB - MODIFIED ORIGINAL CODE
# TOP_BOUNDARY_COLOR      = [255, 255, 0]  # ?? SCAFFOLDING RCB - ORIGINAL CODE

## SKY
SKY_THRESHOLD = 2.0  # 1.85 # 1.5  # ?? SCAFFOLDING RCB - MODIFIED ORIGINAL CODE
# SKY_THRESHOLD = 1.5  # ?? SCAFFOLDING RCB - ORIGINAL CODE
SKY_COLOR = [255, 0, 0]  # BGR - OpenCV's convention

## EDGES
EDGE_COLOR = [255, 255, 255]  # ?? SCAFFOLDING RCB -- ORIGINAL VALUE
# EDGE_COLOR = [255, 0, 0]  # ?? SCAFFOLDING RCB -- TEMPORARY

## CORNERS AND CENTER COLORs
PLT_TOP_LEFT_COLOR = "b"
PLT_TOP_RIGHT_COLOR = "m"
PLT_BOTTOM_RIGHT_COLOR = "g"
PLT_BOTTOM_LEFT_COLOR = "y"
PLT_CENTER_COLOR = "c"

## MAGIC NUMBERS
INTER_POINT_DISTANCE = 20  # for the corners  # ?? SCAFFOLDING RCB - MODIFIED ORIGINAL CODE
SIDE_FACET_DISTANCE = 800  # for the facets  # ?? SCAFFOLDING RCB - MODIFIED ORIGINAL CODE
COMPONENT_THRESHOLD = 35  # 50 # 100 # ?? SCAFFOLDING RCB - MODIFIED ORIGINAL CODE
CLUSTERED_CORNERS_DISTANCE = 100  # ?? SCAFFOLDING RCB - MODIFIED ORIGINAL CODE
# INTER_POINT_DISTANCE        = 20    # for the corners  # ?? SCAFFOLDING RCB - ORIGINAL CODE
# SIDE_FACET_DISTANCE         = 800   # for the facets  # ?? SCAFFOLDING RCB - ORIGINAL CODE
# COMPONENT_THRESHOLD         = 100  # ?? SCAFFOLDING RCB - ORIGINAL CODE
# CLUSTERED_CORNERS_DISTANCE  = 100  # ?? SCAFFOLDING RCB - ORIGINAL CODE

## Tracking
MINIMUM_FRACTION_OF_CONFIRMED_CORNERS = 0.7  # This is a measure of quality of match.  For whatever corners are expected
# to be seen inside the frame, this fraction must match via image confirmation.

MINIMUM_CORNERS_REQUIRED_INSIDE_FRAME = (
    0.7  # This is a measure of sufficiency of information.  This defines the minimum
)
# corners that must be in the heliostat frame to make an adequate determination
# of match.  It is expressed as a ratio because the number of corners per
# heliostat varies with heliostat design.

CLOCKWISE_DIR = ["top_left", "top_right", "bottom_right", "bottom_left"]
# Heliostat
CENTERED_FACET = 13
TOP_LEFT_FACET_INDX = 1
TOP_RIGHT_FACET_INDX = 5
BOTTOM_RIGHT_FACET_INDX = 25
BOTTOM_LEFT_FACET_INDX = 21

TOP_LEFT_CORNER_INDX = 0
TOP_RIGHT_CORNER_INDX = 1
BOTTOM_RIGHT_CORNER_INDX = 2
BOTTOM_LEFT_CORNER_INDX = 3

control_frame_processing = {
    "edges": True,
    "sky": True,
    "facet_boundaries": True,
    "connected_components": True,
    "filt_components": True,
    "fitted_lines_components": True,
    "fitted_lines_inliers_components": True,
    "corners": True,
    "facets": True,
    "clustered_facets": True,
    "projected_corners": True,
}

COEFF_KEYS = [
    ["top_left", "top_edge_coeff"],
    ["top_left", "left_edge_coeff"],
    ["top_right", "top_edge_coeff"],
    ["top_right", "right_edge_coeff"],
    ["bottom_right", "right_edge_coeff"],
    ["bottom_right", "bottom_edge_coeff"],
    ["bottom_left", "left_edge_coeff"],
    ["bottom_left", "bottom_edge_coeff"],
]


# MAX_ROW_DEFAULT = 4320 #2160
# MAX_COL_DEFAULT = 7680 #3840
MAX_ROW_DEFAULT = 2160  # ?? SCAFFOLDING RCB -- USED TO BE SETUP FOR SONY
MAX_COL_DEFAULT = 3840  # ?? SCAFFOLDING RCB -- USED TO BE SETUP FOR SONY
MAXIMUM_TRACK_FRAMES_DEFAULT = 150
## Connected Components

## Inliers connected components
INLIERS_THRESHOLD = 0.7
MIN_TOLERANCE = 0.5
MAX_TOLERANCE = 2.5
TOL_STEP = 0.1


def save_image(img, imgname, path):
    print("In save_image(), saving:", os.path.join(path, imgname))  # ?? SCAFFOLDING RCB -- TEMPORARY
    if img is None:  # ?? SCAFFOLDING RCB -- TEMPORARY
        print(
            "WARNING: In save_image(), img==None encountered for output:", os.path.join(path, imgname)
        )  # ?? SCAFFOLDING RCB -- TEMPORARY
    if img is not None:
        cv.imwrite(os.path.join(path, imgname), img)


def save_fig(img=None, imgname=None, path=None, dpi=500, rgb=False):
    print("In save_fig(),   saving:", os.path.join(path, imgname))  # ?? SCAFFOLDING RCB -- TEMPORARY
    if img is None:  # ?? SCAFFOLDING RCB -- TEMPORARY
        print(
            "WARNING: In save_fig(), img==None encountered for output:", os.path.join(path, imgname)
        )  # ?? SCAFFOLDING RCB -- TEMPORARY
    if img is not None:
        plt.figure()
        if not rgb:
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        else:
            plt.imshow(img)
        plt.savefig(os.path.join(path, imgname), dpi=dpi)
        plt.close()


def sky_with_hsv(img, rgb=False):
    if not rgb:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    light_sky = (100, 30, 100)  # (50, 10, 100)
    dark_sky = (150, 140, 255)  # (150, 140, 255)
    sky = cv.inRange(hsv_img, light_sky, dark_sky)
    sky_img = cv.bitwise_and(img, img, mask=sky)
    return sky, sky_img


def CannyImg(img=None, canny_type="normal", lower=None, upper=None):
    if lower is not None and upper is not None:
        return cv.Canny(img, threshold1=lower, threshold2=upper)
    else:
        if canny_type == "tight":
            lower, upper = 150, 200
        elif canny_type == "normal":
            median_val = np.median(img)
            lower = int(max(0, 0.7 * median_val))
            upper = int(min(255, 1.3 * median_val))
        elif canny_type == "medium":
            lower, upper = 50, 100
        elif canny_type == "light":
            lower, upper = 25, 50
        elif canny_type == "lighter":
            lower, upper = 10, 25

        elif canny_type == "auto":
            v = np.median(img)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))

    return cv.Canny(img, threshold1=lower, threshold2=upper)


def extract_all_frames_from_video(vidoe_path=None, video_name=None, saving_path=None, fps=30):
    cmd = "ffmpeg"
    cmd += " -i " + vidoe_path + video_name + " -vf fps=" + str(fps) + " " + saving_path + "img%d.png"
    subprocess.call(cmd.split())


def extract_frames_from_video(
    video_path=None, video_name=None, saving_path=None, fps=30, starting_time=None, duration=None, backward=False
):
    cmd = "ffmpeg"
    if starting_time is not None and starting_time != "":
        time_instance_parts = starting_time.split(":")
        hrs = time_instance_parts[0]
        minutes = time_instance_parts[1]
        secs = time_instance_parts[2]
        if duration is not None:
            duration_time = "00:00:0" + str(duration)
            if backward:
                minutes_prev = minutes
                if int(secs) - duration + 1 < 0:
                    minutes_prev = int(minutes) - 1
                    secs_prev = 60 + int(secs) - duration + 1
                else:
                    secs_prev = int(secs) - duration + 1
                if secs_prev < 10:
                    previous_time = "00:" + str(minutes_prev) + ":0" + str(secs_prev)
                else:
                    previous_time = "00:" + str(minutes_prev) + ":" + str(secs_prev)
                cmd += " -ss " + previous_time + " -t " + str(duration_time)

                starting_id = (int(minutes_prev) * 60 + int(secs_prev)) * fps
            else:
                starting_id = (int(minutes) * 60 + int(secs)) * fps
                cmd += " -ss " + starting_time + " -t " + str(duration_time)

            cmd += (
                " -i "
                + video_path
                + video_name
                + " -vf fps="
                + str(fps)
                + " "
                + "-start_number "
                + str(starting_id)
                + " "
                + saving_path
                + "/img%d.png"
            )
        else:
            if int(secs) == 59:
                minutes = int(minutes) + 1
                secs = "00"
            else:
                secs = int(secs) + 1
            if len(str(secs)) == 1:
                secs = "0" + str(secs)
            if len(str(minutes)) == 1:
                minutes = "0" + str(minutes)

            time_instance_parts = [str(hrs), str(minutes), str(secs)]
            next_time_instance = ":".join(time_instance_parts)

            cmd += (
                " -i " + video_path + video_name + ".mp4"
                " -ss "
                + starting_time
                + " -to "
                + next_time_instance
                + " -vf fps="
                + str(fps)
                + " "
                + saving_path
                + "img%d.png"
            )

    subprocess.call(cmd.split())


def extract_frames_nopipe(
    video_path=None,
    video_name=None,
    starting_frame_id=None,
    ending_frame_id=None,
    fps=30,
    saving_path=None,
    height=2160,
    width=3840,
):
    saving_path += "img%d.png"
    command = [
        "ffmpeg",
        "-nostats",
        "-loglevel",
        "0",
        "-ss",
        str(starting_frame_id / fps),
        "-i",
        video_path + video_name + ".mp4",
        "-to",
        str(ending_frame_id / fps),
        "-vf",
        str(fps),
        str(saving_path),
        # '-c:v', 'ffv1',
        # #'-vf', 'select=eq(n\,' + str(frame_id) + ')',
        # #'-vsync', '0',
        # '-f', 'image2pipe',
        # '-pix_fmt', 'rgb24',
        # '-vcodec', 'rawvideo', '-'
    ]
    subprocess.call(command)


def extract_frames_opencv(video=None, start=None, end=None, saving_path=None, every=1, plot=False, store=False):
    vid = cv.VideoCapture(video)
    vid.set(cv.CAP_PROP_POS_FRAMES, start - 1)
    frame = start
    saved_count = 0
    while_safety = 0
    frames = []
    while frame < end:
        _, image = vid.read()
        if while_safety > 500:  # break the while if our safety maxs out at 500
            break
        if image is None:
            while_safety += 1
            continue
        if frame % every == 0:
            while_safety = 0  # reset the safety count
            saved_count += 1
            if store:
                imgname = saving_path + "img" + str(saved_count) + ".png"
                if not os.path.exists(imgname):
                    if plot:
                        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                        plt.figure()
                        plt.imshow(image)
                        plt.savefig(imgname)
                        plt.close()
                    else:
                        cv.imwrite(imgname, image)
            frames.append(image)
        frame += 1
    return frames


def extract_specific_frames_pipe(
    video_path=None, video_name=None, starting_frame_id=None, ending_frame_id=None, fps=30, height=2160, width=3840
):
    command = [
        "ffmpeg",
        "-nostats",
        "-loglevel",
        "0",
        "-ss",
        str(starting_frame_id / fps),
        "-i",
        video_path + video_name + ".mp4",
        "-to",
        str(ending_frame_id / fps),
        "-c:v",
        "ffv1",
        #'-vf', 'select=eq(n\,' + str(frame_id) + ')',
        #'-vsync', '0',
        "-f",
        "image2pipe",
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "rawvideo",
        "-",
    ]
    name = str(starting_frame_id) + "_" + str(ending_frame_id) + ".txt"
    name = home_dir() + name
    f1 = open(name, "wb")
    f2 = open(name, "rb")
    pipe = subprocess.Popen(command, stdout=f1, bufsize=10 * 8)
    nbytes = height * width * 3
    image_list = []
    num_frames = ending_frame_id - starting_frame_id
    image = None
    while num_frames > 0:
        frame = f2.read(nbytes)
        if frame is not None:
            image = np.frombuffer(frame, dtype="uint8")
        if image is not None and image.size:
            image = image.reshape((height, width, 3))
            image_list.append(image)
        num_frames -= 1
    # frame = pipe.stdout.read(nbytes)
    # image =  np.frombuffer(frame, dtype='uint8')
    # image = image.reshape((height,width,3))
    os.remove(name)

    return image_list

    # while num_frames > 0:
    # try:
    #     frame  = pipe.stdout.read(nbytes)
    # except IOError:
    #     print ('[No more data]')
    #     break
    # if frame is not None and len(frame):
    #     image =  np.frombuffer(frame, dtype='uint8')
    #     if image.size:
    #         image = image.reshape((height,width,3))
    #         image_list.append(image)
    #         num_frames -= 1
    # else:
    #     break
    # frame = pipe.stdout.read(nbytes)
    # image =  np.frombuffer(frame, dtype='uint8')
    # image = image.reshape((height,width,3))
    pipe.stdout.flush()
    return image_list


def set_proper_hom_coef_sign(point_on_line, btype, A, B, C) -> tuple[float, float, float]:
    """Ensures negative sign distance for points on the mirror side of the line.

    For a given x and y in "Ax + Bx + C", ensure that the sum is positive/negative
    depending on whether xy falls within the mirror boundary (*).

    Returns
    -------
        A, B, and C, possibly with signs flipped"""
    col_on_line, row_on_line = point_on_line[0], point_on_line[1]
    if btype == "left":
        col_to_check, row_to_check = col_on_line + 1, row_on_line

    elif btype == "top":
        col_to_check, row_to_check = col_on_line, row_on_line + 1

    elif btype == "right":
        col_to_check, row_to_check = col_on_line - 1, row_on_line

    elif btype == "bottom":
        col_to_check, row_to_check = col_on_line, row_on_line - 1

    if A * col_to_check + B * row_to_check + C > 0:
        # multiply by -1
        A, B, C = -A, -B, -C

    return A, B, C


def fit_line_component(component=None, type_fit="regression", plot_fit=False):
    """Does a line fit on the pixels in the given component.

    Adds the keys:
       'original_line_hom_coef' : [A, B, C] from Ax + By + C = 0
       'original_line_residual' : whatever the np.polyfit regression residual is
       'original_line_points'   : [x1, y1, x2, y2]
    Notes:
        Chooses a scaling factor such that A^2 + B^2 = 1"""

    def plot_line(pixels, points, hom_coef, color="r"):
        row, col = np.array([a[0] for a in pixels]), np.array([a[1] for a in pixels])
        A, B, C = hom_coef[0], hom_coef[1], hom_coef[2]
        col1, col2 = points[0], points[2]
        x_s = np.linspace(col1, col2, num=50, endpoint=True)
        y_s = (-A * x_s - C) / B
        plt.figure()
        plt.scatter(col, row, label="Pixels", s=1)
        plt.plot(x_s, y_s, color, label="Regression Line", linewidth=1)
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.axis("equal")
        plt.legend()
        plt.show()

    pixels = component["original_pixels"]
    btype = component["boundary_type"]
    if type_fit == "regression":
        row, col = np.array([a[0] for a in pixels]), np.array([a[1] for a in pixels])
        if btype == "left" or btype == "right":
            # expected horizontal line in terms of row
            x, y = row, col
        else:
            # expected horizontal line in terms of col
            x, y = col, row

        reg_fit = np.polyfit(x, y, deg=1, full=True)
        m, b = reg_fit[0]
        residual = reg_fit[1][0]
        # homogeneous coefficients
        A, B, C = -m, 1, -b
        # normalize
        norm = np.linalg.norm(np.array([A, B]))
        A, B, C = A / norm, B / norm, C / norm
        # line points
        x1 = np.min(x)
        y1 = (-A * x1 - C) / B
        x2 = np.max(x)
        y2 = (-A * x2 - C) / B
        if btype == "left" or btype == "right":
            # transpose x, y
            A, B = B, A
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        # setting proper homogeneous coeficients
        start_point = [x1, y1]  # col, row, col, row
        A, B, C = set_proper_hom_coef_sign(start_point, btype, A, B, C)
        # storing to dict
        component["original_line_hom_coef"] = [A, B, C]
        component["original_line_residual"] = residual
        component["original_line_points"] = [x1, y1, x2, y2]  # col, row, col, row
        if plot_fit:
            plot_line(
                pixels=pixels, points=component["original_line_points"], hom_coef=component["original_line_hom_coef"]
            )

    return component


def fit_line_pixels(pixels):
    cols, rows = uncouple_points(pixels)
    diff_col = abs(min(cols) - max(cols))
    diff_row = abs(min(rows) - max(rows))
    if diff_col > diff_row:  # top, bottom
        x, y = cols, rows
        flag = 0
    else:  # left, right
        x, y = rows, cols
        flag = 1

    line = np.polyfit(x, y, deg=1, full=True)
    m, b = line[0]
    A, B, C = -m, 1, -b
    norm = np.linalg.norm(np.array([A, B]))
    A, B, C = A / norm, B / norm, C / norm
    if flag:
        A, B = B, A

    return A, B, C


def fit_line_inliers_pixels(pixels, coeff, min_tolerance=0.5, max_tolerance=5, tol_step=0.1):
    A, B, C = coeff
    required_inliers = int(round(0.7 * len(pixels)))
    tolerance = min_tolerance
    inliers = []
    while tolerance <= max_tolerance:
        inliers = []
        inliers_cnt = 0
        for pixel in pixels:
            col, row = pixel
            if abs(A * col + B * row + C) <= tolerance:
                inliers_cnt += 1
                inliers.append(pixel)

        if inliers_cnt >= required_inliers:
            break
        tolerance += tol_step

    if tolerance <= max_tolerance:
        A, B, C = fit_line_pixels(inliers)

    return A, B, C


def find_hom_line_2points(pt1, pt2):
    col1, row1 = pt1
    col2, row2 = pt2
    diff_col = abs(col1 - col2)
    diff_row = abs(row1 - row2)
    if diff_col > diff_row:  # top, bottom
        x1, x2 = col1, col2
        y1, y2 = row1, row2
        flag = 0
    else:  # left, right
        x1, x2 = row1, row2
        y1, y2 = col1, col2
        flag = 1

    if x2 - x1 == 0:
        return None, None, None

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    A, B, C = -m, 1, -b
    norm = np.linalg.norm(np.array([A, B]))
    A, B, C = A / norm, B / norm, C / norm
    if flag:
        A, B = B, A
    return A, B, C


def min_max_col_row(img, pt1, pt2, epsilon=1):
    MAX_ROW = img.shape[0] - 1
    MAX_COL = img.shape[1] - 1
    col1, row1 = pt1
    col2, row2 = pt2
    min_col = floor(min(col1, col2)) - epsilon
    min_col = max(min_col, 0)
    max_col = ceil(max(col1, col2)) + epsilon
    max_col = min(max_col, MAX_COL)
    min_row = floor(min(row1, row2)) - epsilon
    min_row = max(min_row, 0)
    max_row = ceil(max(row1, row2)) + epsilon
    max_row = min(max_row, MAX_ROW)

    return min_col, max_col, min_row, max_row


def findIntersectionLines(line1, line2):
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    if A1 == 1.0:
        x = -C1 / A1
        y = (-A2 / B2) * x - C2 / B2
    elif A2 == 1.0:
        x = -C2 / A2
        y = (-A1 / B1) * x - C1 / B1
    else:
        x = (C1 / B1 - C2 / B2) / (A2 / B2 - A1 / B1)
        y = (-A1 / B1) * x - C1 / B1

    return [x, y]


def euclidean_distance(point0, point1):
    return sqrt((point0[0] - point1[0]) ** 2 + (point0[1] - point1[1]) ** 2)


def intersection_point(x1, y1, x2, y2, x3, y3, x4, y4) -> list[float]:
    """Given two lines (by the points that define them), return the one xy point where they intersect."""
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    return [px, py]


def avg_pixel(point0, point1):
    col = (point0[0] + point1[0]) / 2.0
    row = (point0[1] + point1[1]) / 2.0
    pixel = [col, row]
    return pixel


def solvePNP(points3d, points2d, h, w, pnptype="pnp", cam_matrix=None, dist_coeff=None):
    # Check input.
    if len(points3d) != len(points2d):
        msg = (
            "In solvePNP(), len(points3d)=" + str(len(points3d)) + " does not equal len(points3d)=" + str(len(points3d))
        )
        print("ERROR: " + msg)
        raise ValueError(msg)
    if len(points3d) < 4:
        msg = "In solvePNP(), len(points3d)=" + str(len(points3d)) + " is not at least 4."
        print("ERROR: " + msg)
        raise ValueError(msg)
    points3d_plane = points3d.copy()
    points3d_plane[:, 2] = 0
    if pnptype == "calib":
        _, mtx, dist, _, _ = cv.calibrateCamera([points3d_plane], [points2d], (w, h), None, None)
        _, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            [points3d], [points2d], (w, h), mtx, dist, flags=cv.CALIB_USE_INTRINSIC_GUESS
        )
        rvec, tvec = rvecs[0], tvecs[0]
    else:
        mtx, dist = cam_matrix, dist_coeff
        if mtx is None:
            mtx = CameraMatrix  # provided camera model
        if dist is None:
            dist = DistCoefs  # provided distortion coefficients
        if pnptype == "pnp":
            _, rvec, tvec = cv.solvePnP(points3d, points2d, mtx, dist)
        elif pnptype == "pnpransac":
            _, rvec, tvec, _ = cv.solvePnPRansac(points3d, points2d, mtx, dist)

    # projection error
    projpoints, _ = cv.projectPoints(points3d, rvec, tvec, mtx, dist)
    projpoints = projpoints.reshape(-1, 2)
    error = cv.norm(points2d, projpoints, cv.NORM_L2) / len(projpoints)
    return mtx, dist, rvec, tvec, error


def uncouple_points(corners):
    x = []
    y = []
    for corner in corners:
        if corner is not None:
            x.append(corner[0])
            y.append(corner[1])
    return x, y


def setup_loger(name, log_file, level=log.INFO):
    formatter = log.Formatter("%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s")
    handler = log.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = log.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def multiprocessing_loger(log_file, level=log.INFO):
    logger = mp.get_logger()
    logger.setLevel(level)
    formatter = log.Formatter("[%(asctime)s| %(levelname)s| %(processName)s] %(message)s")
    handler = log.FileHandler(log_file)
    handler.setFormatter(formatter)

    # this bit will make sure you won't have
    # duplicated messages in the output
    if not len(logger.handlers):
        logger.addHandler(handler)
    return logger


def instance_to_frameid(instance, fps=30):
    secs = int(instance.split(":")[-1])
    mins = int(instance.split(":")[-2])
    total_secs = mins * 60 + secs
    starting_id = total_secs * fps - 1
    ending_id = starting_id + fps
    return starting_id, ending_id


def frameid_to_string(id, frameXd="000000"):
    frameXd = list(frameXd)
    id = str(id)
    j = len(frameXd) - 1
    for i in range(len(id) - 1, -1, -1):
        frameXd[j] = id[i]
        j -= 1

    return "".join(frameXd)
