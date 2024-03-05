import cv2 as cv
import math
import numpy as np
import numpy.typing as npt
import os
import pickle
import shutil
from typing import Callable

import opencsp.common.lib.geometry.angle as angle
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlText as rct
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.system_tools as st

class OpticalFlow():
    _max_cache_count = 10

    def __init__(self,
                 frame1_dir: str, frame1_name_ext: str, frame2_dir: str, frame2_name_ext: str,
                 grayscale_normalization: Callable[[np.ndarray], np.ndarray] = None,
                 # Parameters for sparse optical flow
                 #prevPts, nextPts, status, err, sparse_winSize=None, maxLevel=3, criteria=None, sparse_flags=0, minEigThreshold=1e-4,
                 # Parameters for dense optical flow
                 prev_flow=None, pyr_scale=0.5, levels=5, dense_winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, dense_flags=0, cache=False):
        """ Wrapper class around cv::calcOpticalFlowFarneback (and also cv::calcOpticalFlowPyrLK, eventually).

        Note: opencsp is not compatible with the multiprocessing library on linux. Typical error message::
            "global /io/opencv/modules/core/src/parallel_impl.cpp (240) WorkerThread 6: Can't spawn new thread: res = 11"

        This is due to some sort of bug with how multiprocessing processes and opencv threads interact.
        Possible solutions:
         - use concurrent.futures.ThreadPoolExecutor
         - Loky multiprocessing https://github.com/joblib/loky (I (BGB) couldn't make this one work)

        Args:
        -----
            - frame1_dir (str): Directory for frame1
            - frame1_name_ext (str): First input image, which will be the reference point for the flow
            - frame2_dir (str): Directory for frame2
            - frame2_name_ext (str): Second input image, to compare to the first
            - prev_flow (_type_, optional): I think this is the previous flow calculations to make computation faster? Idk... Defaults to None.
            - pyr_scale (float, optional): parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one. Defaults to 0.5.
            - levels (int, optional): number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used. Defaults to 5.
            - dense_winsize (int, optional): averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field. Defaults to 15.
            - iterations (int, optional): number of iterations the algorithm does at each pyramid level. Defaults to 3.
            - poly_n (int, optional): size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7. Defaults to 5.
            - poly_sigma (float, optional): standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5. Defaults to 1.2.
            - dense_flags (int, optional): operation flags that can be a combination of the following:
                - OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
                - OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize×winsize filter instead of a box filter of the same size for optical flow estimation; usually, this option gives z more accurate flow than with a box filter, at the cost of lower speed; normally, winsize for a Gaussian window should be set to a larger value to achieve the same level of robustness.
            - cache (bool, optional): If True, then pickle the results from the previous 5 computations and save them in the user's home directory. If False, then don't save them. Defaults to False.
                                      The cache option should not be used in production runs. I (BGB) use it for rapid development. It will error when used while running in production (aka on solo).
        """
        self._frame1_dir = frame1_dir
        self._frame1_name_ext = frame1_name_ext
        self._frame2_dir = frame2_dir
        self._frame2_name_ext = frame2_name_ext
        self._grayscale_normalization = grayscale_normalization

        self._prev_flow = prev_flow
        self._pyr_scale = pyr_scale
        self._levels = levels
        self._dense_winsize = dense_winsize
        self._iterations = iterations
        self._poly_n = poly_n
        self._poly_sigma = poly_sigma
        self._dense_flags = dense_flags
        self._cache = cache

        self._mag: np.ndarray = None
        """ XY Matrix. The raw magnitude values returned by opencv, one value per pixel in frame1 """
        self._ang: np.ndarray = None
        """ XY Matrix. The raw angle values returned by opencv, one value per pixel in frame1 """
        self.mag: np.ndarray = None
        """ XY Matrix. The OpenCSP version of the magnitude matrix, where each value corresponds to a pixel in frame1 and
        represents the number of pixels that pixel has moved by frame2. """
        self.ang: np.ndarray = None
        """ XY Matrix. The OpenCSP version of the angle matrix, where each value corresponds to a pixel in frame1 and
        represents the angle that the pixel moved in to frame2 (radians, 0 to the right, positive counter-clockwise). """
        self._cache_dir = os.path.join(orp.opencsp_cache_dir(), "optical_flow")

        if st.is_production_run():
            if cache:
                lt.error_and_raise(ValueError, "OpticalFlow(cache=True) should not be used in production code!")
    
    def clear_cache(self):
        if ft.directory_exists(self._cache_dir):
            ft.delete_files_in_directory(self._cache_dir, "cache*")

    def _get_existing_cache_file(self):
        if not self._cache:
            return None, -1
        ft.create_directories_if_necessary(self._cache_dir)

        # check the previous cache files to see if they match
        for i in range(self.__class__._max_cache_count):
            cache_txt_file = os.path.join(self._cache_dir, f"cache{i}.txt")
            cache_dat_file = os.path.join(self._cache_dir, f"cache{i}.pickle")
            if ft.file_exists(cache_txt_file):
                lines = [l.strip() for l in ft.read_text_file(cache_txt_file)]
                if lines[0] == self._frame1_dir and lines[1] == self._frame1_name_ext and \
                   lines[2] == self._frame2_dir and lines[3] == self._frame2_name_ext and \
                   lines[4] == str(self._grayscale_normalization == None):
                    return cache_dat_file, i
            
        return None, -1

    def _load_from_cache(self):
        prev_dat_file, prev_idx = self._get_existing_cache_file()
        if prev_dat_file != None:
            with open(prev_dat_file, "rb") as fin:
                self._mag, self._ang = pickle.load(fin)
                return True
        return False

    def _save_to_cache(self):
        if not self._cache:
            return

        # remove the existing cache file
        _, prev_idx = self._get_existing_cache_file()
        if prev_idx != -1:
            cache_txt_file = os.path.join(self._cache_dir, f"cache{prev_idx}.txt")
            cache_dat_file = os.path.join(self._cache_dir, f"cache{prev_idx}.pickle")
            ft.delete_file(cache_txt_file, error_on_not_exists=False)
            ft.delete_file(cache_dat_file, error_on_not_exists=False)
        
        # shuffle previous files down the stack
        stop = prev_idx if prev_idx > -1 else self.__class__._max_cache_count
        stop = stop-1
        for i in range(stop,-1,-1):
            cache_txt_file = os.path.join(self._cache_dir, f"cache{i}.txt")
            cache_dat_file = os.path.join(self._cache_dir, f"cache{i}.pickle")
            if i == 4:
                if ft.file_exists(cache_txt_file):
                    ft.delete_file(cache_txt_file, error_on_not_exists=False)
                    ft.delete_file(cache_dat_file, error_on_not_exists=False)
            else:
                cache_txt_file_down = os.path.join(self._cache_dir, f"cache{i+1}.txt")
                cache_dat_file_down = os.path.join(self._cache_dir, f"cache{i+1}.pickle")
                if ft.file_exists(cache_txt_file):
                    shutil.move(cache_txt_file, cache_txt_file_down)
                    shutil.move(cache_dat_file, cache_dat_file_down)
        
        # save to the cache
        cache_txt_file = os.path.join(self._cache_dir, f"cache0.txt")
        cache_dat_file = os.path.join(self._cache_dir, f"cache0.pickle")
        with open(cache_txt_file, "w") as fout:
            fout.write("\n".join([
                self._frame1_dir, self._frame1_name_ext,
                self._frame2_dir, self._frame2_name_ext,
                str(self._grayscale_normalization == None)
            ]))
        with open(cache_dat_file, "wb") as fout:
            pickle.dump((self._mag, self._ang), fout)

    def _load_image(self, f1_or_f2=1) -> npt.NDArray[np.int_]:
        """ Loads either frame 1 or frame 2, converts to grayscale, and applies normalization.
        
        Returns:
        --------
            frame: [np.ndarray] An opencv matrix shape=(h,w) dtype=uint8 """
        if f1_or_f2 == 1:
            path_name_ext_frame = os.path.join(self._frame1_dir, self._frame1_name_ext)
        else:
            path_name_ext_frame = os.path.join(self._frame2_dir, self._frame2_name_ext)
        
        # verify the image exists
        if not ft.file_exists(path_name_ext_frame):
            lt.error_and_raise(FileNotFoundError, f"Error: in OpticalFlow._load_image(), Can't find the frame file \"{path_name_ext_frame}\" for frame {f1_or_f2}")

        # load
        img = cv.imread(path_name_ext_frame)

        # convert to grayscale
        if (len(img.shape) > 2 and img.shape[2] == 3):
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
        # normalize the image colors
        if self._grayscale_normalization != None:
            img = self._grayscale_normalization(img)
        
        return img

    def dense(self) -> tuple[np.ndarray,np.ndarray]:
        """ Computes the optical flow between two images on a pixel-by-pixel basis using the Gunnar Farneback's algorithm.
        https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
        
        Returns:
        --------
            - np.ndarray: self.mag, the magnitude of the flow per pixel (pixels)
            - np.ndarray: self.ang, the direction of the flow per pixel (radians, 0 to the right, positive counter-clockwise) """
        if self._mag is None: # use "is", with "==" numpy does an element-wise comparison
            if not self._load_from_cache():
                # load the images
                frame1 = self._load_image(1)
                frame2 = self._load_image(2)

                # compute the flow
                flow = cv.calcOpticalFlowFarneback(frame1, frame2, self._prev_flow,
                                                self._pyr_scale, self._levels, self._dense_winsize, self._iterations,
                                                self._poly_n, self._poly_sigma, self._dense_flags)
                self._mag, self._ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

                # Computing flow takes a long time, and BGB wants to iterate quickly on a small set of images on his laptop.
                # Cache these results.
                self._save_to_cache()
        
        # magnitude is correct
        self.mag = np.array(self._mag)

        # opencv returns an angle where 0 is on the positive x axis, increasing clockwise
        # we want the angle to increase counter-clockwise
        self.ang = np.array(-self._ang) + (2*np.pi)
            
        return self.mag, self.ang
    
    def limit_by_magnitude(self, lower: float, upper: float, keep="inside"):
        """ Sets any magnitudes not in the keep range (and the corresponding angles) to 0.
        
        Once applied, you can run dense() again to recover the original values.
         
        Arguments:
        ----------
            - lower (float): The bottom of the range of values to include.
            - upper (float): The top of the range of values to include.
            - keep (str): Either "inside" or "outside". If "inside", then values <lower or >upper will
                be set to 0. If "outside", then values >lower and <upper will be set to 0.
                
        Returns:
        --------
            np.ndarray: The indicies that were set to 0. """
        if lower > upper:
            lt.error_and_raise(RuntimeError, f"Error: in OpticalFlow.limit_by_magnitude: lower ({lower}) must be less than upper ({upper})!")

        if keep == "inside":
            bad_indicies = (self.mag < lower) | (self.mag > upper)
        else:
            bad_indicies = (self.mag > lower) & (self.mag < upper)
        self.mag[bad_indicies] = 0
        self.ang[bad_indicies] = 0

        return bad_indicies
    
    def limit_by_angle(self, lower: float, upper: float, keep="inside"):
        """ Same as limit_by_magnitude, but for angle instead. """
        if lower > upper:
            lt.error_and_raise(RuntimeError, f"Error: in OpticalFlow.limit_by_magnitude: lower ({lower}) must be less than upper ({upper})!")
        if upper - lower > np.pi*2:
            lt.error_and_raise(RuntimeError, f"Error: in OpticalFlow.limit_by_magnitude: lower ({lower}) must be within 2pi of upper ({upper})!")

        if lower < 0:
            lower = angle.normalize(lower)
        upper = angle.normalize(upper)
        if lower > upper:
            keep = "outside" if keep == "inside" else "inside"
            lower, upper = upper, lower

        if keep == "inside":
            bad_indicies = (self.ang < lower) | (self.ang > upper)
        else:
            bad_indicies = (self.ang > lower) & (self.ang < upper)
        self.mag[bad_indicies] = 0
        self.ang[bad_indicies] = 0

        return bad_indicies
    
    def to_img(self, mag_render_clip: tuple[float,float]=None):
        """ Converts the flow to an image by mapping:
            - magnitude of movement to value/intensity
            - angle of movement to hue
                - up: green
                - down: red
                - left: blue
                - right: yellow
        
        Note: must call dense() first.

        Args:
        -----
            mag_render_clip (tuple[float,float]): If set, then clip the rendered magnitude to this range (aka low->0, high->255). Defaults to None.

        Returns:
        --------
            np.ndarray: The resulting image, which can be passed to View3D.draw_image (row major, RGB color channels)

        Raises:
        -------
            RuntimeError: If dense() hasn't been executed yet
        """
        # HSV range is 0-179 (H), 0-255 (S), 0-255 (V)
        # https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html
        if not isinstance(self.mag, np.ndarray):
            raise RuntimeError("Error: in OpticalFlow.to_img: self.mag is not an np.ndarray, must call dense() first!")

        h, w = self.mag.shape
        hsv = np.zeros(shape=(h,w,3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        hsv[..., 0] = (self.ang / (2*np.pi) * 179).astype(int)

        if mag_render_clip != None:
            clip_low, clip_high = min(mag_render_clip), max(mag_render_clip)
            render_range = clip_high - clip_low
            min_mag, max_mag = np.min(self.mag), np.max(self.mag)
            hsv = hsv.astype(float)

            # clip minimum and maximum values to stay within render limits
            mag_tmp = np.array(self.mag).clip(clip_low, clip_high)
            hsv[..., 2] = cv.normalize(mag_tmp, None, 0, 255, cv.NORM_MINMAX)

            # clamp values between 0 and 255
            hsv[..., 2] = np.round(hsv[..., 2]).clip(0, 255)
            hsv = hsv.astype(np.uint8)
        
        else:
            hsv[..., 2] = cv.normalize(self.mag, None, 0, 255, cv.NORM_MINMAX)

        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return bgr
    
    def draw_flow_angle_reference(self):
        r = 500
        h = r*2
        w = r*2

        # initialize the circle image, magnitude image, and radians image
        circle_hsv = np.zeros(shape=(h,w,3), dtype=np.uint8)
        mag = np.zeros(shape=(h,w), dtype=np.float32)
        ang = np.zeros(shape=(h,w), dtype=np.float32)

        # generate the magnitudes and colors
        for x in range(w):
            for y in range(h):
                x_dist = abs(x-r)
                y_dist = abs(y-r)

                dist = math.sqrt(x_dist**2 + y_dist**2)
                if dist > r:
                    dist = 0

                if x == r:
                    if y < r:
                        rad = np.pi * 1 / 2
                    else:
                        rad = np.pi * 3 / 2
                else:
                    if x_dist == 0:
                        rad = np.pi * 1 / 2
                    else:
                        rad = math.atan(y_dist / x_dist)
                    if x > r:
                        if y < r: # first quadrant
                            pass
                        else: # fourth quadrant
                            rad = (np.pi * 1 / 2) - rad + (np.pi * 3 / 2)
                    else:
                        if y < r: # second quadrant
                            rad = (np.pi * 1 / 2) - rad + (np.pi * 1 / 2)
                        else: # third quadrant
                            rad = rad + (np.pi * 2 / 2)
                
                mag[y][x] = dist
                ang[y][x] = rad
        
        # convert to HSV space
        circle_hsv[..., 1] = 255
        circle_hsv[..., 0] = (ang / (2*np.pi) * 179).astype(int)
        circle_hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # convert to RGB space
        circle_rgb = cv.cvtColor(circle_hsv, cv.COLOR_HSV2BGR)

        # set values outside the circle to all white
        for x in range(w):
            for y in range(h):
                x_dist = abs(x-r)
                y_dist = abs(y-r)
                dist = math.sqrt(x_dist**2 + y_dist**2)
                if dist > r:
                    circle_rgb[y][x][...] = 255
    
        # copy image onto a larger image, so that we can add text
        r2 = r + 100
        square_rgb = np.zeros_like(circle_rgb, shape=(r2*2,r2*2,3))
        square_rgb.fill(255)
        square_rgb[100:100+w,100:100+h,:] = circle_rgb

        # plot it!
        axis_control = rca.image(grid=False)
        figure_control = rcfg.RenderControlFigure()
        view_spec_2d = vs.view_spec_xy()
        fig_record = fm.setup_figure(figure_control, axis_control, view_spec_2d, title="Optical Flow Reference", code_tag=f"{__file__}", equal=False)
        fig_record.view.draw_image(square_rgb)

        fig_record.view.draw_pq_text((1,0.5), "0", style=rct.RenderControlText(color='k', fontsize=20, horizontalalignment='right'))
        fig_record.view.draw_pq_text((0.5,1), "π/2", style=rct.RenderControlText(color='k', fontsize=20, verticalalignment='top'))
        fig_record.view.draw_pq_text((0,0.5), "π", style=rct.RenderControlText(color='k', fontsize=20, horizontalalignment='left'))
        fig_record.view.draw_pq_text((0.5,0), "3π/2", style=rct.RenderControlText(color='k', fontsize=20, verticalalignment='bottom'))
        
        ang = 0
        prev_ang = ang
        while (ang := ang + 0.2) < np.pi*2:
            x = r2 + (r+30)*math.cos(ang)
            y = r2 + (r+30)*math.sin(ang)
            x = x/(2*r2)
            y = y/(2*r2)
            sang = "%.1f" % ang
            if math.floor(ang+0.01) == math.floor(prev_ang+0.01):
                sang = "." + sang.split(".")[1]
            else:
                sang = "%d" % int(ang)
            fig_record.view.draw_pq_text((x,y), sang, style=rct.RenderControlText(color='k', fontsize=20))
            prev_ang = ang

        fig_record.view.show(block=True)

    @staticmethod
    def get_save_file_name_ext(frame1_name_maybe_ext: str):
        """ Get the file name used to save the results from this
        flow analysis in save(), given the frame1 name. """
        return frame1_name_maybe_ext + "_optflow.npy"

    def _default_save_file_name_ext(self, name_ext=""):
        if name_ext == "":
            name_ext = self._frame1_name_ext + "_optflow.npy"
        return name_ext

    def save(self, dir: str, name_ext="", overwrite=False):
        """ Saves the magnitude and angle matrices computed in the dense() method to the given file.

        Note that this saves the matrices exactly as they were computed in dense (aka
        does not save them as after limit_by_magnitude or limit_by_angle are applied).

        Arguments:
        ----------
            dir (str): The directory to save to.
            name_ext (str, optional): The file to save to. Defaults to get_save_file_name_ext().
            overwrite (bool, optional): True to overwrite the existing optical flow save file. Defaults to False.
        
        Returns:
        --------
            saved_path_name_ext (str): The location for the saved file.
        """
        # I (BGB) tried the save+load with different styles of numpy saving, including
        # np.save(), np.savez(), and np.savez_compressed(). The results for a
        # 7680x4320 image with a NVME drive are:
        #                  | saving | loading | size(MB)
        # save             |  0.98s | 0.10s   | 265
        # savez            |  1.03s | 0.27s   | 265 
        # savez_compressed | 14.5s  | 1.75s   | 210
        name_ext = self._default_save_file_name_ext(name_ext)
        dir_name_ext = os.path.join(dir, name_ext)

        # sanity check
        if self._mag is None: # "is" instead of "==" to avoid np.ndarray element-wise comparison
            lt.error_and_raise(RuntimeError, "Error: in OpticalFlow.save: unable to save non-existant matrices 'magnitude' and 'angle'. Run dense() first!")

        # create the directory as necessary
        ft.create_directories_if_necessary(dir)

        # delete the existing file as necessary
        if ft.file_exists(dir_name_ext):
            if overwrite:
                ft.delete_file(dir_name_ext)
            else:
                lt.error_and_raise(RuntimeError, f"Error: in OpticalFlow.save: unable to save to file \"{dir_name_ext}\", file already exists!")
        
        # save!
        with open(dir_name_ext, "wb") as fout:
            np.save(fout, self._mag, allow_pickle=False)
            np.save(fout, self._ang, allow_pickle=False)

        return dir_name_ext

    @classmethod
    def from_file(cls, dir: str, name_ext: str, error_on_not_exist=True):
        ret = cls("", "a", "", "b")
        ret.load(dir, name_ext, error_on_not_exist)
        return ret
    
    def load(self, dir: str, name_ext: str="", error_on_not_exist=True):
        """ Loads the magnitude and angle matrices from the given file into this instance.

        Args:
            dir (str): The directory to save to.
            name_ext (str, optional): The file to save to. Defaults to self.frame1_name_ext+"_optflow.npy".
            error_on_not_exist (bool, optional): True to throw an error when the given file doesn't exist. Defaults to True.

        Returns:
            mag (np.ndarray): The magnitudes matrix, as is returned from the dense() method, or None if the file doesn't exist
            ang (np.ndarray): The angles matrix, as is returned from the dense() method, or None if the file doesn't exist
        """
        lt.info(f"Loading flow from {dir}/{name_ext}")
        name_ext = self._default_save_file_name_ext(name_ext)
        dir_name_ext = os.path.join(dir, name_ext)

        # check that the file exists
        if not ft.file_exists(dir_name_ext):
            if error_on_not_exist:
                lt.error_and_raise(RuntimeError, f"Error: in OpticalFlow.load: file \"{dir_name_ext}\" doesn't exist!")
            return None, None
        
        # load!
        with open(dir_name_ext, "rb") as fin:
            self._mag: np.ndarray = np.load(fin, allow_pickle=False)
            self._ang: np.ndarray = np.load(fin, allow_pickle=False)
            self.dense() # populate self.mag and self.dense
        
        return self.mag, self.ang
