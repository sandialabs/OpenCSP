import numpy as np

from opencsp.app.sofast.lib.ProcessSofastFixed import ProcessSofastFixed
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
import opencsp.common.lib.tool.log_tools as lt


def measurement_stats(sofast: ProcessSofastFringe | ProcessSofastFixed) -> list[dict]:
    """Returns statistics as list of dictionaries per each given facet"""

    if isinstance(sofast, ProcessSofastFringe):
        num_facets = sofast.num_facets
    elif isinstance(sofast, ProcessSofastFixed):
        num_facets = 1

    stats = []

    for idx_facet in range(num_facets):
        if isinstance(sofast, ProcessSofastFringe):
            # Get data
            data_calc = sofast.data_characterization_facet[idx_facet]
            data_im_proc = sofast.data_image_processing_facet[idx_facet]
            data_surf = sofast.data_surfaces[idx_facet]

            # Sample resolution
            mask = data_im_proc.mask_processed
            im_x = np.zeros(mask.shape) * np.nan
            im_y = np.zeros(mask.shape) * np.nan
            im_x[mask] = data_calc.v_surf_points_facet.x
            im_y[mask] = data_calc.v_surf_points_facet.y

            # Number of points
            num_samps = len(data_calc.v_surf_points_facet)
        elif isinstance(sofast, ProcessSofastFixed):
            # Get data
            data_surf = sofast.slope_solver.surface
            data_calc = sofast.data_slope_solver
            # Sample resolution
            surf_points = sofast.data_slope_solver.v_surf_points_facet
            pts_index_xy = sofast.blob_index.get_data()[1]
            point_indices_mat = sofast.blob_index.get_data_mat()[1]
            offset_x = sofast.blob_index._offset_x
            offset_y = sofast.blob_index._offset_y
            im_x = np.zeros(point_indices_mat.shape[:2]) * np.nan
            im_y = np.zeros(point_indices_mat.shape[:2]) * np.nan
            im_y[pts_index_xy.y - offset_y, pts_index_xy.x - offset_x] = surf_points.y
            im_x[pts_index_xy.y - offset_y, pts_index_xy.x - offset_x] = surf_points.x
            # Number of points
            num_samps = len(surf_points)
        else:
            lt.error_and_raise(ValueError, f'Input type, {type(sofast)} is not supported.')

        # Calculate average sample resolution
        dx = np.diff(im_x, axis=1)  # meters
        dy = np.diff(im_y, axis=0)  # meters
        dx_avg = abs(np.nanmean(dx))  # meters
        dy_avg = abs(np.nanmean(dy))  # meters

        # Parabolic focal length in x and y
        if isinstance(data_surf, Surface2DParabolic):
            surf_coefs = data_calc.surf_coefs_facet
            focal_lengths_xy = [1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]]
        else:
            focal_lengths_xy = [np.nan, np.nan]

        stats.append(
            {
                'delta_x_sample_points_average': dx_avg,
                'delta_y_sample_average': dy_avg,
                'number_samples': num_samps,
                'focal_lengths_parabolic_xy': focal_lengths_xy,
            }
        )

    return stats
