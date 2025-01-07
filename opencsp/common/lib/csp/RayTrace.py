import os
import queue
import threading
import time
from functools import reduce
from multiprocessing.pool import Pool
from typing import Iterable
from warnings import warn

import numpy as np
import psutil
from scipy.spatial.transform import Rotation

from opencsp.common.lib.csp import LightPath as lp
from opencsp.common.lib.csp.LightPath import LightPath
from opencsp.common.lib.csp.LightPathEnsemble import LightPathEnsemble
from opencsp.common.lib.csp.LightSource import LightSource
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.geometry.RegionXY import Resolution
import opencsp.common.lib.csp.Scene as scn
from opencsp.common.lib.geometry.FunctionXYGrid import FunctionXYGrid
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlRayTrace import RenderControlRayTrace
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets, save_hdf5_datasets
from opencsp.common.lib.tool.typing_tools import strict_types


class RayTrace:
    """
    A class for performing ray tracing in a given scene.

    This class manages the ray tracing process, including the collection of light paths
    and the rendering of the scene based on the light sources and objects present.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(self, scene: scn.Scene = None) -> None:
        """
        Initializes a RayTrace object with the specified scene.

        Parameters
        ----------
        scene : scn.Scene, optional
            The scene in which to perform ray tracing (default is None).
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        if scene == None:
            scene = scn.Scene()
        self.scene = scene
        self.light_paths_ensemble = LightPathEnsemble([])
        self.save_file_location = None
        pass

    @property
    def light_paths(self):
        """
        Retrieves the list of light paths in the ray trace.

        Returns
        -------
        list[LightPath]
            A list of LightPath objects representing the light paths in the scene.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return self.light_paths_ensemble.asLightPathList()

    def __str__(self) -> str:
        for lp in self.light_paths_ensemble:
            print(lp)

    def __add__(self, trace: 'RayTrace'):
        sum_trace = RayTrace()

        for light_source in self.scene.light_sources + trace.scene.light_sources:
            sum_trace.scene.add_light_source(light_source)

        for obj in self.scene.objects + trace.scene.objects:
            sum_trace.scene.add_object(obj)

        sum_trace.light_paths_ensemble = self.light_paths_ensemble + trace.light_paths_ensemble

        return sum_trace

    def ray_count(self) -> int:
        """
        Returns the number of light paths in the ray trace.

        Returns
        -------
        int
            The number of light paths in the ensemble.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return len(self.light_paths_ensemble)

    def draw(self, view: View3d, trace_style: RenderControlRayTrace = None) -> None:
        """
        Draws the light paths in a 3D view.

        Parameters
        ----------
        view : View3d
            The 3D view in which to draw the light paths.
        trace_style : RenderControlRayTrace, optional
            The style settings for rendering the light paths (default is None).

        Returns
        -------
        None
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        if trace_style == None:
            trace_style = RenderControlRayTrace()
        for lp in self.light_paths_ensemble:
            lp.draw(view, trace_style.light_path_control)

    def draw_subset(self, view: View3d, count: int, trace_style: RenderControlRayTrace = None):
        """
        Draws a subset of light paths in a 3D view.

        Parameters
        ----------
        view : View3d
            The 3D view in which to draw the light paths.
        count : int
            The number of light paths to draw.
        trace_style : RenderControlRayTrace, optional
            The style settings for rendering the light paths (default is None).

        Returns
        -------
        None
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        for i in np.floor(np.linspace(0, len(self.light_paths_ensemble) - 1, count)):
            lp = self.light_paths_ensemble[int(i)]
            lp.draw(view, trace_style.light_path_control)

    # @strict_types
    def add_many_light_paths(self, new_paths: list[LightPath]):
        """
        Adds multiple light paths to the ray trace.

        Parameters
        ----------
        new_paths : list[LightPath]
            A list of LightPath objects to add to the ray trace.

        Returns
        -------
        None
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        self.light_paths_ensemble.concatenate_in_place(LightPathEnsemble(new_paths))

    @classmethod
    def from_hdf(cls, filename: str, trace_name: str = "RayTrace") -> 'RayTrace':
        """
        Creates a RayTrace object from an HDF5 file.

        Parameters
        ----------
        filename : str
            The path to the HDF5 file containing the ray trace data.
        trace_name : str, optional
            The name of the trace to load from the file (default is "RayTrace").

        Returns
        -------
        RayTrace
            A RayTrace object initialized with data from the specified HDF5 file.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        trace = RayTrace()

        batch_names: list[str] = list(load_hdf5_datasets([f"RayTrace_{trace_name}/Batches"], filename).values())[0]
        lpe = LightPathEnsemble([])
        for batch in batch_names:
            prefix = f"RayTrace_{trace_name}/Batches/{batch}/"
            subgroups = [prefix + 'CurrentDirections', prefix + 'InitialDirections', prefix + 'Points']
            curr_directions, init_directions, points = list(load_hdf5_datasets(subgroups, filename).values())
            curr_directions = Uxyz(curr_directions)
            init_directions = Uxyz(init_directions)
            points = list(map(Pxyz, points))
            lpe.concatenate_in_place(LightPathEnsemble.from_parts(init_directions, points, curr_directions))

        trace.light_paths_ensemble = lpe
        return trace

    pass  # end of class


def calc_reflected_ray(normal_v: Vxyz, incoming_v: Vxyz) -> Vxyz:
    """
    Calculates the direction of the reflected ray given the incident ray direction and surface normal.

    Parameters
    ----------
    normal_v : Vxyz
        The normal vector at the surface point where the reflection occurs.
    incoming_v : Vxyz
        The direction of the incoming ray.

    Returns
    -------
    Vxyz
        The direction of the reflected ray.

    Notes
    -----
    The algorithm for reflection is based on the formula:

    .. code-block:: text

        \[
        \text{reflected\_ray} = \text{incoming\_ray} - 2 \cdot (\text{normal} \cdot \text{incoming\_ray}) \cdot \text{normal}
        \]

        norm_v and inc_v must broadcast together.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    # Process input vector
    n = normal_v.normalize().data
    V0 = incoming_v.normalize().data

    # Compute reflected ray direction
    ref_vecs = V0 - 2 * np.matmul(n, np.matmul(n.T, V0))

    return Vxyz(ref_vecs)


def process_vector(vec: np.ndarray, norm: bool = False) -> np.ndarray:
    """
    Reshapes and converts input vectors to floats.

    Parameters
    ----------
    vec : array-like, shape (3, N) or (3,)
        Input vectors to be processed.
    norm : bool, optional
        If True, the vector will be normalized (default is False).

    Returns
    -------
    np.ndarray
        A 3xN array of floats, normalized if specified.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    # Reshape and convert to float
    vec = np.array(vec).astype(float)
    vec = vec.reshape((3, -1))

    # Normalize
    if norm:
        return vec / np.linalg.norm(vec)

    return vec


def trace_scene_unvec(
    scene: scn.Scene,
    obj_resolution: int,
    random_dist: bool = False,
    store_in_ram: bool = True,
    save_in_file: bool = False,
    save_name: str = f"ray_trace_{time.asctime().replace(' ','_').replace(':','_')}",
    verbose: bool = False,
) -> RayTrace:
    """
    DEPRECATED: Traces rays through the scene using an unvectorized approach.

    Parameters
    ----------
    scene : scn.Scene
        The scene to trace rays through.
    obj_resolution : int
        The resolution for sampling points on objects.
    random_dist : bool, optional
        If True, random distribution of rays will be used (default is False).
    store_in_ram : bool, optional
        If True, the results will be stored in RAM (default is True).
    save_in_file : bool, optional
        If True, the results will be saved to a file (default is False).
    save_name : str, optional
        The name of the file to save the results (default is a timestamped name).
    verbose : bool, optional
        If True, prints updates during processing (default is False).

    Returns
    -------
    RayTrace
        A RayTrace object containing the traced rays.

    Raises
    ------
    DeprecationWarning
        If this function is called, indicating it is deprecated.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    warn(
        "RayTrace.trace_scene is vectorized and will be faster. This function will be phased out.",
        DeprecationWarning,
        stacklevel=2,
    )

    ray_trace = RayTrace(scene)
    for obj in scene.objects:
        # get the points on the mirror to reflect off of
        points_and_normals = obj.survey_of_points(obj_resolution)
        # from tutorial https://www.geeksforgeeks.org/python-unzip-a-list-of-tuples/
        unzipped_p_and_n = list(zip(*points_and_normals))
        just_points = unzipped_p_and_n[0]
        just_normals = unzipped_p_and_n[1]

        # TODO TJL:vectorized version of the raytrace algorithm, we should look at upsides before implemting.
        # for num, (p, n_v) in enumerate(points_and_normals): # point, normal vector # TODO TJL:verify output is correct
        #     rays = LightPath.many_rays_from_many_vectors([], np.transpose(incoming_vectors)) # make rays from points and vectors
        #     normal_vector = n_v
        #     ref_vec = calc_reflected_ray(normal_vector, incoming_vectors)
        #     for i, ray in enumerate(rays):
        #         ray.add_step(p, ref_vec[:, i])
        #     print(f"{num}/{tot} through tracing")

        # TODO TJL:unvectorized version, this could be optimal with parallelization or we might need to finish the vectorized version above.
        tot = len(points_and_normals)

        if verbose:
            tot = len(points_and_normals)
            checkpoints = [int(np.ceil(tot * n / 10)) for n in range(10)]

        # for i, (p, n_v) in enumerate(points_and_normals): # loop through points
        #     inc_vs = []
        #     for ls in scene.light_sources: # loop through light sources
        #         for incoming_light in ls.get_incident_rays(p): # get the rays from the light source for the current point
        #             inc_vs.append(lp.normalize(incoming_light.current_direction))
        #     ref_vec = calc_reflected_ray(n_v, inc_vs)
        #     ray = LightPath.many_rays_from_many_vectors([p for _ in range()], )
        #     ray.add_step(p, ref_vec[:, 0])
        #     ray_trace.add_light_path(ray)
        #     if verbose and i in checkpoints: # TODO TJL:make sure this check does not slow down the program
        #         print(f"{i/tot*100}% through tracing.")

        for i, (p, n_v) in enumerate(points_and_normals):  # loop through points
            for ls in scene.light_sources:  # loop through light sources
                # get the rays from the light source for the current point
                for incoming_light in ls.get_incident_rays(p):
                    vector_from_path = lp.normalize(incoming_light.current_direction)
                    ref_vec = calc_reflected_ray(n_v, vector_from_path)
                    ray = LightPath([], vector_from_path)
                    ray.add_step(p, ref_vec[:, 0])
                    ray_trace.add_light_path(ray)
            if verbose and i in checkpoints:  # TODO TJL:make sure this check does not slow down the program
                print(f"{i/tot:.2%} through tracing.")

        # if save_in_file:
        #     ray_trace.save_trace(save_name)
        #     if verbose:
        #         print(f"Ray Trace saved as {save_name}.")

    return ray_trace


# TODO TJL:FIX ISSUES, only trace_scene_parallel is up-to-date
# # @strict_types
def trace_scene(
    scene: scn.Scene,
    obj_resolution: Resolution,
    store_in_ram: bool = True,
    save_in_file: bool = False,
    save_name: str = None,
    trace_name: str = "Default",
    max_ram_in_use_percent: float = 99,
    verbose: bool = False,
) -> RayTrace:
    """
    Traces rays through the scene using a vectorized approach.

    Parameters
    ----------
    scene : scn.Scene
        The scene to trace rays through.
    obj_resolution : Resolution
        The resolution for sampling points on objects.
    store_in_ram : bool, optional
        If True, the results will be stored in RAM (default is True).
    save_in_file : bool, optional
        If True, the results will be saved to a file (default is False).
    save_name : str, optional
        The name of the file to save the results (default is None).
    trace_name : str, optional
        The name of the trace for saving (default is "Default").
    max_ram_in_use_percent : float, optional
        The maximum percentage of RAM to use during tracing (default is 99).
    verbose : bool, optional
        If True, prints updates during processing (default is False).

    Returns
    -------
    RayTrace
        A RayTrace object containing the traced rays.

    Raises
    ------
    UserWarning
        If saving is requested but the flag is set to False.
    ValueError
        If saving is requested but no file name is provided.
    MemoryError
        If the maximum RAM usage is exceeded before tracing begins.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    # argument validity checks
    if not save_in_file and save_name != None:
        warn(
            "Saving file was specified, but 'save_in_file' flag is set to False. Trace will not be saved.",
            UserWarning,
            stacklevel=2,
        )
    if save_in_file and save_name == None:
        raise ValueError("save_in_file flag was True, but no file was specified to dave in.")
    if save_in_file and max_ram_in_use_percent < psutil.virtual_memory().percent:
        raise MemoryError("Maximum memory allocated to ray trace was reached before the trace has begun")

    # start
    if verbose:
        print("Setting up Ray Trace...")

    ray_trace = RayTrace(scene)
    ray_trace.save_file_location = save_name

    total_lpe = LightPathEnsemble([])
    batch = int(0)

    # # @strict_types
    def trace_for_single_object(obj: RayTraceable) -> LightPathEnsemble:

        total_lpe: LightPathEnsemble = LightPathEnsemble([])

        if verbose:
            print("getting survey...")

        # Get the points on the mirror to reflect off
        points, normals = obj.survey_of_points(obj_resolution)

        if verbose:
            print("...got survey")

        if verbose:
            number_of_rays = len(points)
            checkpoints = [int(np.ceil(number_of_rays * n / 10)) for n in range(1, 10)]
            print("Beginning Ray Trace...")

        # Loop through points and perform trace calculations
        for i, (p, n_v) in enumerate(zip(points, normals)):

            ################# TODO TJL:draft for saving traces ######################
            if save_in_file and max_ram_in_use_percent < psutil.virtual_memory().percent:
                prefix = f"RayTrace_{trace_name}/Batches/Batch{batch:03}/"
                datasets = [prefix + "InitialDirections", prefix + "Points", prefix + "CurrentDirections"]
                data = [
                    total_lpe.init_directions.data,
                    np.array([points.data for points in total_lpe.points_lists]),
                    total_lpe.current_directions.data,
                ]
                if verbose:
                    print("saving...")
                save_hdf5_datasets(data, datasets, save_name)
                total_lpe = LightPathEnsemble([])
                if verbose:
                    print(f"Batch {batch} is over, now we start batch {(batch:=batch+1)}")
            ##############################################################################

            p: Pxyz
            n_v: Vxyz
            lps: list[LightPath] = []

            # concatenates all incoming rays from all light sources
            for ls in scene.light_sources:
                lps += ls.get_incident_rays(p)
            lpe = LightPathEnsemble(lps)

            ################ ray trace algorithm ################
            P = np.array([list(p.data)] * len(lpe)).T
            V0 = lpe.current_directions.data
            N = np.array(n_v.data).T
            results = V0 - 2 * np.matmul(N.T, np.matmul(N, V0))
            #####################################################

            # add to the LightPathEnsemble
            lpe.add_steps(Pxyz(P), Uxyz(results))
            total_lpe.concatenate_in_place(lpe)

            if verbose and i in checkpoints:  # TODO TJL:make sure this check does not slow down the program
                print(
                    f"{i/number_of_rays:.2%} through tracing. Using {psutil.virtual_memory().percent}% of system RAM."
                )

        # if the user wants to store the result in ran add to this RayTrace object
        if store_in_ram:
            ray_trace.light_paths_ensemble += total_lpe

        return total_lpe

    for obj in scene.objects:
        trace_for_single_object(obj)

    # Save Last Batch
    if save_in_file:
        prefix = f"RayTrace/Batches/Batch{batch:08}/"
        datasets = [prefix + "InitialDirections", prefix + "Points", prefix + "CurrentDirections"]
        data = [
            total_lpe.init_directions.data,
            np.array([points.data for points in total_lpe.points_lists]),
            total_lpe.current_directions.data,
        ]
        print("saving...")
        save_hdf5_datasets(data, datasets, save_name)

    return ray_trace


# Helper for trace_scene_parallel
# # @strict_types
def _trace_object(
    process: int,
    obj: RayTraceable,
    obj_resolution: Resolution,
    resolution_type: str,
    verbose: bool,
    light_sources: list[LightSource],
    #   store_in_ram: bool = True,
    #   hdf_filename: str = None,  # None means it will not save
    #   trace_name: str = "Default",
) -> tuple[int, LightPathEnsemble]:

    total_lpe: LightPathEnsemble = LightPathEnsemble([])
    batch = 0

    # Get the points on the mirror to reflect off
    if verbose:
        print(f"Process #{process:03}: batch {batch:03} getting survey...")  # TODO
    points, normals = obj.survey_of_points(obj_resolution)
    if verbose:
        print(f"Process #{process:03}: ...batch {batch:03} got survey")  # TODO

    if verbose:
        number_of_rays = len(points)
        checkpoints = [int(np.ceil(number_of_rays * n / 10)) for n in range(1, 10)]
        print(f"Process #{process:03}: Batch #{batch:03} Beginning Ray Trace...")

    # Loop through points and perform trace calculations
    for i, (p, n_v) in enumerate(zip(points, normals)):

        # ###################### TODO TJL:draft for saving traces ######################
        # if save_in_file and max_ram_in_use_percent < psutil.virtual_memory().percent:
        #     prefix = f"RayTrace/Batches/Batch{batch:08}/"
        #     datasets = [
        #         prefix + "InitialDirections",
        #         prefix + "Points",
        #         prefix + "CurrentDirections",
        #     ]
        #     data = [
        #         total_lpe.init_directions.data,
        #         np.array([points.data for points in total_lpe.points_lists]),
        #         total_lpe.current_directions.data
        #     ]
        #     print("saving...")
        #     save_hdf5_datasets(data, datasets, save_name)
        #     total_lpe = LightPathEnsemble([])
        #     if verbose:
        #         print(f"Batch {batch} is over, now we start batch {(batch:=batch+1)}")
        # ###################################################################################

        # type annotations
        p: Pxyz
        n_v: Vxyz
        lps: list[LightPath] = []

        # concatenates all incoming rays from all light sources
        for ls in light_sources:
            lps += ls.get_incident_rays(p)
        lpe = LightPathEnsemble(lps)

        ############################# Ray Trace Algorithm #############################
        V0 = lpe.current_directions.data  # incoming vectors
        N = np.array(n_v.data).T  # normal vector
        results = V0 - 2 * np.matmul(N.T, np.matmul(N, V0))  # trace vectorized
        ###############################################################################

        # add to the LightPathEnsemble
        P = np.array([list(p.data)] * len(lpe)).T
        lpe.add_steps(Pxyz(P), Uxyz(results))
        total_lpe.concatenate_in_place(lpe)

        # if the user wants to store the result in ran add to this RayTrace object

    if verbose:
        print(
            f"Process #{process:03}: Batch #{batch:03} finished. Using {psutil.virtual_memory().percent}% of system RAM."
        )

    # ################################# Save in hdf5 file #################################
    # if hdf_filename != None:
    #     prefix = f"RayTrace_{trace_name}/Batches/Batch_{process:03}_{batch:03}/"
    #     dataset_names = [
    #         prefix + "InitialDirections",
    #         prefix + "Points",
    #         prefix + "CurrentDirections",
    #     ]
    #     ray_trace_data = [
    #         total_lpe.init_directions.data,
    #         np.array([points.data for points in total_lpe.points_lists]),
    #         total_lpe.current_directions.data
    #     ]
    #     save_hdf5_datasets(ray_trace_data, dataset_names, hdf_filename)
    #     # while True:
    #     #     try:
    #     #         save_hdf5_datasets(ray_trace_data, dataset_names, hdf_filename)
    #     #         break
    #     #     except OSError:
    #     #         if verbose:  # TODO TJL:make sure this is not too slow
    #     #             print(f"Process #{process:03}: Failed to save Batch #{batch:03}. Trying again...")
    #     #         time.sleep(0.01)
    #     if verbose:
    #         print(f"Process #{process:03}: Batch #{batch:03} saved.")

    # #####################################################################################

    # if not store_in_ram:
    #     return LightPathEnsemble([])
    return (process, total_lpe)


def trace_scene_parallel(
    scene: scn.Scene,
    obj_resolution: Resolution,
    processor_count: int,
    resolution_type: str = 'pixelX',
    store_in_ram=True,
    max_ram_in_use_percent: float = 99.0,
    save_in_file=False,
    save_file_name: str = None,
    trace_name: str = "RayTrace",
    verbose: bool = False,
) -> RayTrace:
    """
    Traces rays through the scene using parallel processing.

    This function divides the ray tracing task among multiple processes to improve performance.

    Parameters
    ----------
    scene : scn.Scene
        The scene to trace rays through.
    obj_resolution : Resolution
        The resolution for sampling points on objects.
    processor_count : int
        The number of processors to use for parallel processing.
    resolution_type : str, optional
        The type of resolution to use (default is 'pixelX').
    store_in_ram : bool, optional
        If True, the results will be stored in RAM (default is True).
    max_ram_in_use_percent : float, optional
        The maximum percentage of RAM to use during tracing (default is 99.0).
    save_in_file : bool, optional
        If True, the results will be saved to a file (default is False).
    save_file_name : str, optional
        The name of the file to save the results (default is None).
    trace_name : str, optional
        The name of the trace for saving (default is "RayTrace").
    verbose : bool, optional
        If True, prints updates during processing (default is False).

    Returns
    -------
    RayTrace
        A RayTrace object containing the traced rays.

    Raises
    ------
    UserWarning
        If saving is requested but the flag is set to False.
    ValueError
        If saving is requested but no file name is provided.
    MemoryError
        If the maximum RAM usage is exceeded before tracing begins.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    ################# argument validity checks ####################
    # if not save_in_file and save_file_name != None:
    #     warn("Saving file was specified, but 'save_in_file' flag is set to False. Trace will not be saved.",
    #          UserWarning, stacklevel=2)
    # if save_in_file and save_file_name == None:
    #     raise ValueError("save_in_file flag was True, but no file was specified to dave in.")
    # if save_in_file and max_ram_in_use_percent < psutil.virtual_memory().percent:
    #     raise MemoryError("Maximum memory allocated to ray trace was reached before the trace has begun")
    num_of_cpu_available = os.cpu_count()
    if processor_count > num_of_cpu_available:
        warn(
            f"too many processors were allocated for ray trace. There were {processor_count} "
            f"processors allocated, but only {num_of_cpu_available} available. "
            f"Trace will use maximum number available"
        )
        processor_count = num_of_cpu_available
    ###############################################################

    # start
    if verbose:
        print("Setting up Ray Trace...")

    # initializations
    ray_trace = RayTrace(scene)
    ray_trace.save_file_location = save_file_name
    batch = int(0)  # start at batch 0

    # smallest objects that are still RayTraceables
    # this allows for splitting into pieces for parallelization
    if verbose:
        print("Splitting up objects...")

    # collect the most basic objects in the scene (i.e. mirrors)
    basic_objects: RayTraceable = []
    for obj in scene.objects:
        basic_objects += obj.most_basic_ray_tracable_objects()

    if verbose:
        print(f"Found {len(basic_objects)} basic RayTraceables...")

    # multiprocessing
    with Pool(processor_count) as pool:
        if verbose:
            print(f"Pooled {processor_count} processors...")
        count = len(basic_objects)

        # q = queue.Queue()
        # q_thread = threading.Thread(target=_threaded_saving_queue, args=(q, save_file_name, trace_name, verbose))

        trace_map = pool.starmap(
            _trace_object,
            zip(
                range(count),  # process number
                basic_objects,  # objects to trace
                [obj_resolution] * count,  # object relsolution (all the same)
                [resolution_type] * count,  # resolution_type (all the same)
                [verbose] * count,  # verbosity (all the same)
                [scene.light_sources] * count,  # light sources (all the same)
                #  [store_in_ram] * count,            # store_in_ram (all the same)
                #  [save_file_name] * count,          # save_file_name (all the same)
                #  [trace_name] * count,              # trace_name (all the same)
            ),
        )

    # finalize results
    final_lpe = LightPathEnsemble([])
    for process, process_lpe in trace_map:
        ################################# Save in hdf5 file #################################
        if save_file_name != None:
            prefix = f"RayTrace_{trace_name}/Batches/Batch_{process:03}_{batch:03}/"
            dataset_names = [prefix + "InitialDirections", prefix + "Points", prefix + "CurrentDirections"]
            ray_trace_data = [
                process_lpe.init_directions.data,
                np.array([points.data for points in process_lpe.points_lists]),
                process_lpe.current_directions.data,
            ]
            save_hdf5_datasets(ray_trace_data, dataset_names, save_file_name)
            # while True:
            #     try:
            #         save_hdf5_datasets(ray_trace_data, dataset_names, save_file_name)
            #         break
            #     except OSError:
            #         if verbose:  # TODO TJL:make sure this is not too slow
            #             print(f"Process #{process:03}: Failed to save Batch #{batch:03}. Trying again...")
            #         time.sleep(0.01)
            if verbose:
                print(f"Process #{process:03}: Batch #{batch:03} saved.")
        #####################################################################################

        ####################### Collect each process output into one ########################
        if store_in_ram:
            final_lpe.concatenate_in_place(process_lpe)
        #####################################################################################
    if store_in_ram:
        ray_trace.light_paths_ensemble = final_lpe
    return ray_trace


def plane_intersect(
    ray_trace: RayTrace,
    v_plane_center: Vxyz,
    u_plane_norm: Uxyz,
    epsilon: float = 1e-6,
    verbose: bool = False,
    save_in_file: bool = False,
    save_name: str = None,
    max_ram_in_use_percent: float = 95.0,
):
    """
    Finds all intersections that occur at a specified plane from the light paths in the ray trace.

    The output points are transformed from the global reference frame to the local plane reference frame,
    where the XY plane is perpendicular to the target normal.

    Parameters
    ----------
    ray_trace : RayTrace
        The RayTrace object containing the light paths.
    v_plane_center : Vxyz
        The center point of the plane.
    u_plane_norm : Uxyz
        The normal vector of the plane.
    epsilon : float, optional
        The threshold for determining if a ray is parallel to the plane (default is 1e-6).
    verbose : bool, optional
        If True, prints execution status (default is False).
    save_in_file : bool, optional
        If True, saves the intersection data to a file (default is False).
    save_name : str, optional
        The name of the file to save the intersection data (default is None).
    max_ram_in_use_percent : float, optional
        The maximum percentage of RAM to use during processing (default is 95.0).

    Returns
    -------
    Vxy
        The intersection points in the local plane XY reference frame.

    Raises
    ------
    ValueError
        If the input parameters are invalid.
    MemoryError
        If the maximum RAM usage is exceeded during processing.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    # Unpack plane

    intersecting_points = []
    lpe = ray_trace.light_paths_ensemble
    batch = 0

    # ################# TODO TJL:draft for saving traces ######################
    # # TODO: add deletion if save_in_ram = False
    # if save_in_file:
    #     datasets = [
    #         f"Intersection/Information/PlaneLocation",
    #         f"Intersection/Information/PlaneNormalVector",
    #     ]
    #     data = [v_plane_center.data, u_plane_norm.data]
    #     print("saving general information...")
    #     save_hdf5_datasets(data, datasets, save_name)
    # ##############################################################################

    # finds where the light intersects the plane
    # algorithm explained at \opencsp\doc\IntersectionWithPlaneAlgorithm.pdf
    # TODO TJL:upload explicitly vectorized algorithm proof

    u_plane_norm = u_plane_norm.normalize()
    plane_vectorV = u_plane_norm.data  # column vector
    v_plane_centerV = v_plane_center.data  # column vector

    # most recent points in light path ensemble
    if verbose:
        print("setting up values...")
    P = Pxyz.merge(list(map(lambda xs: xs[-1], lpe.points_lists))).data
    V = lpe.current_directions.data  # current vectors

    if verbose:
        print("finding intersections...")

    ########## Intersection Algorithm ###########
    # .op means to do the 'op' element wise
    d = np.matmul(plane_vectorV.T, V)  # (1 x N) <- (1 x 3)(3 x N)
    W = P - v_plane_centerV  # (3 x N) <- (3 x N) -[broadcast] (3 x 1)
    f = -np.matmul(plane_vectorV.T, W) / d  # (1 x N) <- (1 x 3)(3 x N) ./ (1 x N)
    F = f * V  # (3 x N) <- (1 x N) .* (3 x N)
    intersection_matrix = P + F  # (3 x N) <- (3 x N) .- (3 x N)
    #############################################
    intersection_points = Pxyz(intersection_matrix)

    # filter out points that miss the plane
    if verbose:
        print("filtering out missed vectors")
    filtered_intersec_points = Pxyz.merge(list(filter(lambda vec: not vec.hasnan(), intersection_points)))

    if verbose:
        print("Rotating.")
    # TODO: Do we want the inverse that takes that vector back into the up vector
    # up_vec = Vxyz([0, 0, 1])
    # rot = Vxyz.align_to(u_plane_norm, up_vec)  # .inv()
    # rotated_intersec_points: Pxyz = filtered_intersec_points.rotate(rot)

    if verbose:
        print("Plane intersections caluculated.")

    ################# TODO TJL:draft for saving traces ######################
    if save_in_file:
        datasets = [f"Intersection/Batches/Batch{batch:08}"]
        if verbose:
            print(type(filtered_intersec_points.data))
        data = [filtered_intersec_points.data]
        if verbose:
            print(f"saving to {save_name}...")
        save_hdf5_datasets(data, datasets, save_name)
    ##############################################################################

    return filtered_intersec_points.projXY()
    # return np.histogram2d(xyz[:,0], xyz[:,1], bins)
    # TODO TJL:create the histogram from this or bin these results


def histogram_image(bin_res: float, extent: float, pts: Vxy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a 2D histogram from scattered points.

    This function generates a histogram image (point spread function) based on the provided
    points in the XY plane, using specified bin resolution and extent.

    Parameters
    ----------
    bin_res : float
        The resolution of the histogram bins in meters.
    extent : float
        The width of the image area in meters.
    pts : Vxy
        A Vxy object containing the points to calculate the XY histogram.

    Returns
    -------
    hist : np.ndarray
        A 2D array representing the histogram image (point spread function).
    x : np.ndarray
        A 1D array representing the X axis in meters.
    y : np.ndarray
        A 1D array representing the Y axis in meters.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    bins = int(extent / bin_res)
    extent = bin_res * bins
    rng = [[-extent / 2, extent / 2]] * 2

    hist, x, y = np.histogram2d(pts.x, pts.y, range=rng, bins=bins, density=False)  # (y, x)
    hist = hist.T  # (x, y)
    hist = np.flip(hist, 0)  # convert from image to array
    return hist, x, y


def ensquared_energy(pts: Vxy, semi_width_max: float, res: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the ensquared energy as a function of square half-width.

    This function computes the fraction of ensquared energy within a square defined by the
    semi-widths, based on the provided points.

    Parameters
    ----------
    pts : Vxy
        A Vxy object containing the points for which to calculate ensquared energy.
    semi_width_max : float
        The maximum semi-width of the square in meters.
    res : int, optional
        The resolution (number of data points) for the semi-widths (default is 50).

    Returns
    -------
    np.ndarray
        An array containing the fraction of ensquared energy for each semi-width.
    np.ndarray
        An array of semi-widths in meters.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    # Calculate widths
    ws = np.linspace(0, semi_width_max, res)
    fracs = []
    num_pts = float(len(pts))
    for w in ws:
        (x1, x2, y1, y2) = (-w, w, -w, w)
        mask = (pts.x > x1) * (pts.x < x2) * (pts.y > y1) * (pts.y < y2)
        fracs.append(float(mask.sum()) / num_pts)
    return np.array(fracs), ws
