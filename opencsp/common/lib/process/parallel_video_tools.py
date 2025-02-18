import os

import opencsp.common.lib.process.ParallelPartitioner as ppart
import opencsp.common.lib.process.ServerSynchronizer as ss
import opencsp.common.lib.render_control.RenderControlVideo as rcv
import opencsp.common.lib.render_control.RenderControlVideoFrames as rcvf
import opencsp.common.lib.render.VideoHandler as vh
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def parallel_frames_to_videos(
    partitioner: ppart.ParallelPartitioner,
    frames_path: str,
    frames_names: list[str],
    videos_path: str,
    render_control: rcv.RenderControlVideo = None,
    overwrite=False,
):
    """Converts the given frames into videos, one video per computer.

    To convert all of the given frames, this same function should be called
    with the same arguments across all computers.
    Note that there isn't much of a point in parallelizing across cpu cores on
    a single computer because ffmpeg already uses all available cpu cores.

    Example usage::

        # worker function:
        partitioners = ppart.ParallelPartitioner.get_partitioners(nservers, my_server_idx, ncpus=1)
        frames_names = ft.files_in_directory_by_extension(frames_path, ["jpg"], sort=False)["jpg"]
        videos_names = pvt.parallel_frames_to_videos(partitioner, frames_path, frames_names, videos_path)
        with multiprocessing.Pool(ncores) as p:
            videos_names = p.starmap(pvt.parallel_frames_to_videos, [(partitioner, frames_path, frames_names, videos_path) for partitioner in partitioners])
        return list(filter(lambda f: f != None, videos_names))

        # calling function:
        if my_server_idx == 0:
            video_handler.merge_videos(partial_videos_names)

    Args:
    -----
        partitioner (ParallelPartitioner): Helper for portioning data to this server/cpu.
        frames_path (str): What directory to search for frames in.
        frames_suffix (str): The file suffix for the frames. Eg ".jpg"
        videos_path (str): Where to put the generated video(s).
        render_control (RenderControlVideo): Controls the properties of the generated video.

    Returns:
    --------
        str: The generated video file (name and extension). None if frames_names is empty or there are no frames for this cpu core.
    """
    # sanitize inputs
    if not ft.directory_exists(frames_path):
        raise RuntimeError(f"Could not find the frames directory '{frames_path}'!")
    if len(frames_names) == 0:
        return None
    ft.create_directories_if_necessary(videos_path)

    # Get a subselection of the frames for parallel execution
    lt.info(
        f"partition info: [nservers: {partitioner.nservers}, server_idx: {partitioner.server_idx}, ncpus: {partitioner.ncpus}, cpu_idx: {partitioner.cpu_idx}]"
    )
    my_frames = partitioner.get_my_portion(frames_names, "Frame to Videos")
    lt.info(f"Size of my_frames: {len(my_frames)}/{len(frames_names)}")
    if len(my_frames) == 0:
        return None

    # build the video for this server
    video_out_file_ext = f"out_{partitioner.identifier()}.mp4"
    video_out_path_file_ext = os.path.join(frames_path, video_out_file_ext)
    _, _, ext = ft.path_components(frames_names[0])
    frame_control = rcvf.RenderControlVideoFrames(inframe_format=ext)
    video_handler: vh.VideoHandler = vh.VideoHandler.VideoCreator(
        frames_path, video_out_path_file_ext, render_control, frame_control
    )
    video_name_ext = video_handler.frames_to_video(my_frames, overwrite=overwrite)

    return video_name_ext


def parallel_video_to_frames(
    num_servers: int, server_index: int, video_handler: vh.VideoHandler, server_synchronizer: ss.ServerSynchronizer
):
    """Extract all frames from the given video, where each server extracts the frames for part of the video.
    To extract all frames, execute this method on each server with that server's server_index.

    We use the time select method "-ss start_time -to end_time", where each server starts at the
    exact same time that the last server stopped. This way there should be a 1-frame overlap
    between the frames that the servers output.

    There is another method that could be frame-exact, meaning that there should be no (extra)
    duplicate frames beyond what would be already duplicated by ffmpeg running on a single server.
    The way to achieve this is with the "trim=start_frame=n:end_frame=m" option for the
    output stream, where n and m are determined per-server based on the length and framerate
    of the source video.
    However, we don't use that method, because in practice what happens is ffmpeg duplicates the
    first frame n times. So instead we use the likely ok but maybe not time select method.

    Args:
    -----
        - num_servers (int): How many servers this is being evaluated on.
        - server_index (int): Which server out of num_servers this is being evaluated on. Indexing starts at 0.
        - video_handler (VideoHandler): The handler to use to extract the frames
        - server_synchronizer (ServerSynchronizer): The synchronizer for all servers to wait on while collected extracted frames into a single directory.
    """
    src_video_dir_name_ext = video_handler.src_video_dir_name_ext
    dst_frames_dir = video_handler.dst_frames_dir
    dst_example_frames_dir = video_handler.dst_example_frames_dir
    frame_control = video_handler.frame_control

    # build the extraction directories for this server
    dst_frames_dir_serv = os.path.join(dst_frames_dir, f"extraction_server_{server_index}")
    ft.create_directories_if_necessary(dst_frames_dir_serv)
    if dst_example_frames_dir != None:
        dst_example_frames_dir_serv = os.path.join(dst_example_frames_dir, f"extraction_server_{server_index}")
        ft.create_directories_if_necessary(dst_example_frames_dir_serv)
    frame_name_format = frame_control.get_outframe_name(src_video_dir_name_ext, is_example_frames=False)
    frame_name_format_example = frame_control.get_outframe_name(src_video_dir_name_ext, is_example_frames=True)
    video_handler = vh.VideoHandler.VideoExtractor(
        src_video_dir_name_ext, dst_frames_dir_serv, dst_example_frames_dir_serv, frame_control
    )

    # determine the number of frames in the video, and which ones this server should extract
    video_tot_frames = video_handler.get_num_frames()
    video_duration = video_handler.get_duration()
    framerate = video_tot_frames / video_duration
    single_server_range_nframes = int(video_tot_frames / num_servers)
    single_server_range_seconds = single_server_range_nframes / framerate
    rstart = single_server_range_seconds * server_index
    rend = single_server_range_seconds * (server_index + 1)
    lt.info(
        f"In parallel_video_to_frames(), server @{server_index} (0-{num_servers-1}) extracting frames from {rstart:0.1f}s to {rend:0.1f}s"
    )

    # extract the frames!
    video_handler.extract_frames(start_time=rstart, end_time=rend)

    # remove any duplicates
    duplicates_handler = vh.VideoHandler.VideoCreator(dst_frames_dir_serv, None, None, frame_control)
    (non_duplicate_frame_files, duplicate_frame_files) = duplicates_handler.identify_duplicate_frames(0, 0)
    for dup_frame in duplicate_frame_files:
        dup_frame = os.path.join(dst_frames_dir_serv, dup_frame)
        ft.delete_file(dup_frame)
    for fi, src_frame in enumerate(non_duplicate_frame_files):
        src_frame = os.path.join(dst_frames_dir_serv, src_frame)
        dst_frame = os.path.join(dst_frames_dir_serv, frame_name_format % (fi + 1))
        ft.rename_file(src_frame, dst_frame, is_file_check_only=(fi > 0))
    lt.info(
        f"In parallel_video_to_frames(), server extracted {len(non_duplicate_frame_files)} frames and {len(duplicate_frame_files)} duplicates"
    )

    # if server 0, then:
    # a) wait for all the other servers to finish
    # b) collect their extracted frames into a single directory
    server_synchronizer.wait()

    if server_index == 0:
        for dst_dir, fnf in [(dst_frames_dir, frame_name_format), (dst_example_frames_dir, frame_name_format_example)]:
            if dst_dir == None:
                continue

            tot_frames = 0
            for si in range(num_servers):
                dst_dir_serv = os.path.join(dst_dir, f"extraction_server_{si}")
                num_frames_serv = ft.count_items_in_directory(dst_dir_serv)

                # move the non-duplicate frames to the common directory
                for fi in range(1, num_frames_serv + 1):
                    src_frame = os.path.join(dst_dir_serv, fnf % fi)
                    dst_frame = os.path.join(dst_dir, fnf % (tot_frames + 1))
                    # use the rename_file utility for the first file to make sure it got renamed successfully
                    # only check for is_file for the rest of the files because the full check is slow
                    ft.rename_file(src_frame, dst_frame, is_file_check_only=(fi > 1))
                    tot_frames += 1
                lt.info(f"Moved {num_frames_serv} frames from server {si}")

                # check that the server's extraction directory is empty, then remove the directory
                num_residual = ft.count_items_in_directory(dst_dir_serv)
                if num_residual != 0:
                    lt.warn(
                        f"Warning: in parallel_video_to_frames(), server's extraction directory isn't empty! (found {num_residual} files)"
                    )
                else:
                    os.rmdir(dst_dir_serv)

    server_synchronizer.wait()
