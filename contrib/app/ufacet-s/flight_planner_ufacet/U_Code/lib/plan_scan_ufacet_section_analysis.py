"""
Analyze a UFACET flight scanning path within an (x,y) section.



"""

import math
import numpy as np

import opencsp.common.lib.geometry.geometry_2d as g2d
import opencsp.common.lib.render.view_spec as vs


# -------------------------------------------------------------------------------------------------------
# CONSTRAINT ANALYSIS
#


def construct_minimum_safe_altitude_line(section_context, heliostat_name_list):
    # Fetch required parameters.
    solar_field = section_context["solar_field"]
    view_spec = section_context["view_spec"]

    # Construct line connecting first and last heliostat origins.
    first_heliostat = solar_field.lookup_heliostat(heliostat_name_list[0])
    last_heliostat = solar_field.lookup_heliostat(heliostat_name_list[-1])
    first_pq = vs.xyz2pq(first_heliostat.origin, view_spec)
    last_pq = vs.xyz2pq(last_heliostat.origin, view_spec)
    origin_line = g2d.homogeneous_line(first_pq, last_pq)

    # Orient origin line so that (x,y) points will give positive values if above the line.
    if origin_line[1] < 0:
        origin_line = g2d.flip_homogeneous_line(origin_line)

    # Determine whether any heliostats protrude above this line (e.g., due to a hill in the terrain).
    maximum_signed_distance = -math.inf
    for heliostat_name in heliostat_name_list:
        heliostat = solar_field.lookup_heliostat(heliostat_name)
        heliostat_pq = vs.xyz2pq(heliostat.origin, view_spec)
        signed_distance = g2d.homogeneous_line_signed_distance_to_xy(heliostat_pq, origin_line)
        if signed_distance > maximum_signed_distance:
            maximum_signed_distance = signed_distance

    # Construct a line that is above every heliostat.
    above_all_origin_line = origin_line
    above_all_origin_line[2] -= maximum_signed_distance

    # Construct start and end points on the "above all" line.
    first_p = first_pq[0]  # Units are meters.  Defined relative to tower
    corrected_first_q = g2d.homogeneous_line_y_given_x(
        first_p, above_all_origin_line
    )  # base origin.  Heliostat origin is at center
    last_p = last_pq[0]  # of torque tube.
    corrected_last_q = g2d.homogeneous_line_y_given_x(last_p, above_all_origin_line)  #

    # Fetch corner (x,y,z) when heliostat is face up.  Add y-component to torque tube origin z
    # to get maximum height a heliostat can attain.
    mirror_half_height = first_heliostat.top_left_corner_offset[1]  # m.
    # Altitude safety margin.
    altitude_margin = section_context["altitude_margin"]  # meters.
    # Minimum allowable commanded height above heliostat origin.
    mnsa_height_above_origin = mirror_half_height + altitude_margin
    # Construct minimum safe altitude constraint.
    mnsa_ray_tail = [first_p, (corrected_first_q + mnsa_height_above_origin)]
    mnsa_ray_head = [last_p, (corrected_last_q + mnsa_height_above_origin)]
    mnsa_ray = [mnsa_ray_tail, mnsa_ray_head]
    # Construct maximum safe altitude constraint.
    maximum_safe_altitude = section_context["maximum_safe_altitude"]  # meters.
    mxsa_ray_tail = [first_p, maximum_safe_altitude]
    mxsa_ray_head = [last_p, maximum_safe_altitude]
    mxsa_ray = [mxsa_ray_tail, mxsa_ray_head]

    # Save key results.
    section_context["above_all_origin_line"] = above_all_origin_line
    section_context["mnsa_height_above_origin"] = mnsa_height_above_origin
    section_context["mnsa_ray"] = mnsa_ray
    section_context["mxsa_ray"] = mxsa_ray

    # Return.
    return section_context


def construct_flight_path_family(section_context):
    # Fetch required parameters.
    above_all_origin_line = section_context["above_all_origin_line"]
    mnsa_height_above_origin = section_context["mnsa_height_above_origin"]
    maximum_safe_altitude = section_context["maximum_safe_altitude"]

    # Family of flight paths, indexed by "altitude" -- actually distance from origin measured perpendicular to path slope.
    #
    # Flight path line equation.
    #    A is the x component of the average-ground surface normal.
    #    B is the y component of the average-ground surface normal.
    #    C is distance from origin to path line, in meters.  If path is level, z is altitude.
    # The path corresponds to the homogeneous line equation Ax + By + C = 0.
    A = -above_all_origin_line[0]  # Flip orientation, so increasing C implies increasing altitude.
    B = -above_all_origin_line[1]  #
    C = -above_all_origin_line[2]  #
    # The coefficient C is the flight path height above the global (x,y,z) origin.
    # Setting C_min = C would specify a flight path roughly through the heliostat origins.
    # Instead we set C_min to correspond to the first available safe flight altitude.
    C_mnsa = C + mnsa_height_above_origin  # m.
    C_mxsa = maximum_safe_altitude  # m.  Inexact, because this doesn't account for slope.
    C_min = C_mnsa  # m.
    C_max = C_mxsa  # m.  (C_max - C_min) is arbitrary, and simply controls range covereed in computation.
    C_step = 1.0  # m   C_step is arbitrary, and controls altitude resolution of analysis.

    # Save path family.
    section_context["path_family_A"] = A
    section_context["path_family_B"] = B
    section_context["path_family_C_mnsa"] = C_mnsa
    section_context["path_family_C_mxsa"] = C_mxsa
    section_context["path_family_C_min"] = C_min
    section_context["path_family_C_max"] = C_max
    section_context["path_family_C_step"] = C_step

    # Return.
    return section_context


def fetch_key_points(section_context, h_a_name, h_t_name_list, h_b_name):
    # Fetch required parameters.
    solar_field = section_context["solar_field"]
    view_spec = section_context["view_spec"]

    # Assessed heliostat.
    h_a = solar_field.lookup_heliostat(h_a_name)
    h_a_top_xyz = (h_a.top_left_corner + h_a.top_left_corner) / 2.0
    h_a_bottom_xyz = (h_a.bottom_left_corner + h_a.bottom_right_corner) / 2.0
    # Convert to section coordinates.
    at_pq = vs.xyz2pq(h_a_top_xyz, view_spec)
    ab_pq = vs.xyz2pq(h_a_bottom_xyz, view_spec)

    # Target corners.
    t_pq_list = []
    for h_t_name in h_t_name_list:
        # Fetch heliostat.
        h_t = solar_field.lookup_heliostat(h_t_name)
        h_t_top_xyz = (h_t.top_left_corner + h_t.top_right_corner) / 2.0
        # Convert to section coordinates.
        t_pq = vs.xyz2pq(h_t_top_xyz, view_spec)
        t_pq_list.append(t_pq)

    # Background corner.
    if h_b_name:
        h_b = solar_field.lookup_heliostat(h_b_name)
        h_b_bottom_xyz = (h_b.bottom_left_corner + h_b.bottom_right_corner) / 2.0
        # Convert to section coordinates.
        bb_pq = vs.xyz2pq(h_b_bottom_xyz, view_spec)
    else:
        bb_pq = None

    # Return.
    return at_pq, ab_pq, t_pq_list, bb_pq


def construct_reflected_point(mirror_pq, nu, target_pq):
    # Constructs the reflected point of an optical target reflected at
    # the mirror_pq point by a mirror with surface normal direction nu
    # (measured ccw from x-axis).
    dp_mirror_to_target = target_pq[0] - mirror_pq[0]
    dq_mirror_to_target = target_pq[1] - mirror_pq[1]
    theta = math.atan2(dq_mirror_to_target, dp_mirror_to_target)  # Angle ccw from p axis to ray [mirror --> target].
    iota = theta - nu  # Incidence angle.
    rho = nu - iota  # Reflected ray angle from p axis, measured ccw.
    length = 10
    reflected_p = mirror_pq[0] + (length * np.cos(rho))
    reflected_q = mirror_pq[1] + (length * np.sin(rho))
    return [reflected_p, reflected_q]


def ray_eta(ray):
    # Angle of ray, measured ccw from x axis.
    xy_tail = ray[0]
    x_tail = xy_tail[0]
    y_tail = xy_tail[1]
    xy_head = ray[1]
    x_head = xy_head[0]
    y_head = xy_head[1]
    dx = x_head - x_tail
    dy = y_head - y_tail
    return math.atan2(dy, dx)


def single_heliostat_gaze_angle_analysis(section_context, ab_pq, at_pq, s_locus, e_locus, constraints):
    # Fetch family of flight paths.
    A = section_context["path_family_A"]
    B = section_context["path_family_B"]
    C_min = section_context["path_family_C_min"]
    C_max = section_context["path_family_C_max"]
    C_step = section_context["path_family_C_step"]

    # Initialize loop variables.
    path_s_pq_list = []
    path_e_pq_list = []
    ray_min_eta_list = []
    ray_max_eta_list = []
    min_etaC_list = []
    max_etaC_list = []
    C = C_min
    # Loop.
    while C <= C_max:
        path = [A, B, C]
        path_s_pq = g2d.intersect_lines(path, g2d.homogeneous_line(s_locus[0], s_locus[1]))
        path_e_pq = g2d.intersect_lines(path, g2d.homogeneous_line(e_locus[0], e_locus[1]))
        ray_min_eta = [path_e_pq, ab_pq]
        min_eta = ray_eta(ray_min_eta)
        ray_max_eta = [path_s_pq, at_pq]
        max_eta = ray_eta(ray_max_eta)
        min_etaC = [min_eta, C]
        max_etaC = [max_eta, C]
        path_s_pq_list.append(path_s_pq)
        path_e_pq_list.append(path_e_pq)
        ray_min_eta_list.append(ray_min_eta)
        ray_max_eta_list.append(ray_max_eta)
        min_etaC_list.append(min_etaC)
        max_etaC_list.append(max_etaC)
        # Increment loop variable.
        C += C_step

    # Save constraints.
    constraints["path_s_pq_list"] = path_s_pq_list  # Path sacn region start point, as a function of C.
    constraints["path_e_pq_list"] = path_e_pq_list  # Path sacn region start point, as a function of C.
    constraints["ray_min_eta_list"] = (
        ray_min_eta_list  # Ray pointing from path start point to assessed heliostat top edge, function of C.
    )
    constraints["ray_max_eta_list"] = (
        ray_max_eta_list  # Ray pointing from path end point to assessed heliostat bottom edge, function of C.
    )
    constraints["min_etaC_list"] = min_etaC_list  # Lower bound on required gaze angle interval, as a function of C.
    constraints["max_etaC_list"] = max_etaC_list  # Upper bound on required gaze angle interval, as a function of C.

    # Return.
    return constraints


def single_heliostat_gaze_angle_selection(section_context, constraints):
    # Fetch the camera vertical field of view.
    camera = section_context["camera"]
    # For now, ignore zoom lens focal length selection and just assume widest angle setting.
    camera_vertical_fov = camera.fov_vertical_max

    # Fetch the gaze angle lower and uppper required limits, as a function of C.
    min_etaC_list = constraints["min_etaC_list"]
    max_etaC_list = constraints["max_etaC_list"]
    # Check input.
    if len(min_etaC_list) != len(max_etaC_list):
        print(
            "ERROR: In, single_heliostat_gaze_angle_selection(), mismatched min_etaC and max_etaC lengths encountered."
        )
        assert False

    # Shift the gaze angle required limits to obtain constraints on the center gaze angle.
    d_eta = camera_vertical_fov / 2.0  # Shift by half the field of view.
    shifted_min_etaC_list = [[etaC[0] + d_eta, etaC[1]] for etaC in min_etaC_list]
    shifted_max_etaC_list = [[etaC[0] - d_eta, etaC[1]] for etaC in max_etaC_list]

    # Construct the feasible gaze angles.
    envelope_min_etaC_list = []
    envelope_max_etaC_list = []
    for shifted_min_etaC, shifted_max_etaC in zip(shifted_min_etaC_list, shifted_max_etaC_list):
        if shifted_min_etaC[0] >= shifted_max_etaC[0]:
            envelope_min_etaC_list.append(shifted_max_etaC)  # Lines crossed, so we swap min.max sense.
            envelope_max_etaC_list.append(shifted_min_etaC)  #

    # Shrink by gaze tolerance.
    gaze_tolerance = section_context["gaze_tolerance"]
    shrunk_min_etaC_list = []
    shrunk_max_etaC_list = []
    for envelope_min_etaC, envelope_max_etaC in zip(envelope_min_etaC_list, envelope_max_etaC_list):
        C = envelope_min_etaC[1]
        envelope_min_eta = envelope_min_etaC[0]
        envelope_max_eta = envelope_max_etaC[0]
        shrunk_min_eta = envelope_min_eta + gaze_tolerance
        shrunk_max_eta = envelope_max_eta - gaze_tolerance
        if shrunk_min_eta <= shrunk_max_eta:
            shrunk_min_etaC_list.append([shrunk_min_eta, C])
            shrunk_max_etaC_list.append([shrunk_max_eta, C])

    # Clip to eta limits.
    eta_min = section_context["eta_min"]
    eta_max = section_context["eta_max"]
    clipped_min_etaC_list = []
    clipped_max_etaC_list = []
    for envelope_min_etaC, envelope_max_etaC in zip(shrunk_min_etaC_list, shrunk_max_etaC_list):
        C = envelope_min_etaC[1]
        envelope_min_eta = envelope_min_etaC[0]
        envelope_max_eta = envelope_max_etaC[0]
        clipped_min_eta = max(eta_min, envelope_min_eta)
        clipped_max_eta = min(eta_max, envelope_max_eta)
        if clipped_min_eta <= clipped_max_eta:
            clipped_min_etaC_list.append([clipped_min_eta, C])
            clipped_max_etaC_list.append([clipped_max_eta, C])

    # Fork based on desired gaze control.
    gaze_type = section_context["gaze_type"]
    if gaze_type == "constant":
        # Select gaze angle.
        if len(clipped_min_etaC_list) > 0:
            lowest_min_etaC = clipped_min_etaC_list[0]
            lowest_max_etaC = clipped_max_etaC_list[0]
            selected_cacg_eta = (lowest_min_etaC[0] + lowest_max_etaC[0]) / 2.0
            # Select altitude.
            C_clipped = lowest_min_etaC[1]
            C_critical = constraints["C_critical"]
            selected_cacg_C = max(C_clipped, C_critical)
        else:
            print("WARNING: In single_heliostat_gaze_angle_selection(), infeasible case encountered.")
            print("               len(envelope_min_etaC_list) = ", len(envelope_min_etaC_list))
            print("               len(shrunk_min_etaC_list)   = ", len(shrunk_min_etaC_list))
            print("               len(clipped_min_etaC_list)  = ", len(clipped_min_etaC_list))
            # Set values that will force the following code to clip the altitude and gaze angle.
            selected_cacg_eta = np.deg2rad(-200.0)  # Extreme value.
            selected_cacg_C = 100000.0  # m.   Signal value.

        # If result exceeds the maximum altitude, then clip to that alttiude, while preseving maximum eta constraint.
        #
        # This is because the maximum eta constraint ensures that we will see the heliostat first without back-side
        # feature reflection, which provides two benefits:
        #    1. As features sweep up the mirror, we will obtain optical targets all the way from the bottom up.
        #    2. The mirror's first appearance will generally be reflecting sky only, facilitating automatic recognition.
        #    3. We will see the full heliostat reflecting sky, supporting per-facet inspection tasks across all mirrors.
        #
        max_C = section_context["maximum_altitude"]
        if selected_cacg_C > max_C:
            print("NOTE: In single_heliostat_gaze_angle_selection(), clipping to max altitude.")
            # Pick altitude, then lookup eta that ensures mirror will be seen intially facing all sky.
            if len(shifted_max_etaC_list) < 2:
                print(
                    "ERROR: In single_heliostat_gaze_angle_selection(), unexpected short shifted_max_etaC_list encountered."
                )
                assert False
            selected_cacg_C = max_C  # m.
            if shifted_max_etaC_list[0][1] > selected_cacg_C:
                print("ERROR: In single_heliostat_gaze_angle_selection(), shifted_max_etaC_list starts above C.")
                assert False
            # Search for point on eta = f(C) curve.
            selected_cacg_eta = math.inf
            prev_C = -math.inf
            for max_etaC in shifted_max_etaC_list:
                C = max_etaC[1]
                if not (C > prev_C):
                    print(
                        "ERROR: In single_heliostat_gaze_angle_selection(), shifted_max_etaC_list is not in order of ascending C."
                    )
                    assert False
                if C >= selected_cacg_C:
                    selected_cacg_eta = max_etaC[0]
                    break
                prev_C = C
            if selected_cacg_eta == math.inf:
                print("ERROR: In single_heliostat_gaze_angle_selection(), unexpected selected_cacg_eta == math.inf.")
                assert False
            if prev_C > selected_cacg_C:
                print("ERROR: In single_heliostat_gaze_angle_selection(), unexpected prev_C > selected_cacg_C.")
                assert False
            if C < selected_cacg_C:
                print("ERROR: In single_heliostat_gaze_angle_selection(), unexpected C < selected_cacg_C.")
                assert False
            # Shrink.
            # Recall max and min swapped.
            selected_cacg_eta += gaze_tolerance
            # Clip.
            if selected_cacg_eta < eta_min:
                selected_cacg_eta = eta_min
    else:
        print(
            'ERROR: In single_heliostat_gaze_angle_selection(), unexpected gaze_type="'
            + str(gaze_type)
            + '" encountered.'
        )
        assert False

    # Assemble selected (eta, C) pair.
    selected_cacg_etaC = [selected_cacg_eta, selected_cacg_C]

    # Save result.
    constraints["shifted_min_etaC_list"] = shifted_min_etaC_list
    constraints["shifted_max_etaC_list"] = shifted_max_etaC_list
    constraints["envelope_min_etaC_list"] = envelope_min_etaC_list
    constraints["envelope_max_etaC_list"] = envelope_max_etaC_list
    constraints["shrunk_min_etaC_list"] = shrunk_min_etaC_list
    constraints["shrunk_max_etaC_list"] = shrunk_max_etaC_list
    constraints["clipped_min_etaC_list"] = clipped_min_etaC_list
    constraints["clipped_max_etaC_list"] = clipped_max_etaC_list
    constraints["selected_cacg_etaC"] = selected_cacg_etaC  # "cacg" == "constant altitude, constant gaze"

    # Return.
    return constraints


def single_heliostat_section_analysis(
    section_context,
    # Heliostat of interest.
    heliostat_name_list,  # Must be sorted with closest first.
    assess_heliostat_name,
):
    # Control parameters.
    p_margin = section_context["p_margin"]
    maximum_target_lookback = section_context["maximum_target_lookback"]

    # Fetch context parameters.
    mnsa_ray = section_context["mnsa_ray"]

    # Fetch key heliostats.
    # Assessed.
    h_a_name = assess_heliostat_name
    h_a_idx = heliostat_name_list.index(h_a_name)
    n_names = len(heliostat_name_list)
    # Reflection targets.
    h_t_name_list = []
    h_t_idx = h_a_idx - 1
    while (h_t_idx >= 0) and (len(h_t_name_list) < maximum_target_lookback):
        h_t_name_list.append(heliostat_name_list[h_t_idx])
        h_t_idx -= 1
    # Background behind assessed.
    if h_a_idx < (n_names - 1):
        h_b_name = heliostat_name_list[h_a_idx + 1]  # Background corner.
    else:
        h_b_name = None
    at_pq, ab_pq, t_pq_list, bb_pq = fetch_key_points(section_context, h_a_name, h_t_name_list, h_b_name)

    # If there are no target points, createa a fictitious target to simplify downstream computation.
    if len(t_pq_list) == 0:
        if bb_pq == None:
            print(
                "ERROR: In single_heliostat_section_analysis(), No-target, no background case not supported.  Use vanity mode."
            )
            assert False
        dp = at_pq[0] - bb_pq[0]
        fictitous_t_pq = [(ab_pq[0] + dp), at_pq[1]]  # Imitate [assessed --> background] relative postion.
        t_pq_list.append(fictitous_t_pq)

    # Assessed surface normal.
    dp_abat = at_pq[0] - ab_pq[0]
    dq_abat = at_pq[1] - ab_pq[1]
    tau = math.atan2(dq_abat, dp_abat)  # Angle from p axis to assessed heliostat tangent.
    nu = tau + (np.pi / 2)  # Angle from p axis to assessed heliostat surface normal.

    # PASS START ANALYSIS

    # CONSTRAINT: Assessed bottom must be visible.
    # abv_lb = "assessed bottom visibility, lower bound."  A lower bound on p.
    if len(t_pq_list) > 0:
        abv_lb = [ab_pq, t_pq_list[0]]
        abvm_lb = g2d.shift_x(abv_lb, p_margin)
    else:
        abv_lb = None
        abvm_lb = None

    # CONSTRAINT: Assessed top must be clear of background.
    if bb_pq:
        # atv_lb = "assessed top visibility, lower bound."  A lower bound on p.
        atv_lb = [bb_pq, at_pq]
        atvm_lb = g2d.shift_x(atv_lb, p_margin)
    else:
        atv_lb = None
        atvm_lb = None

    # CONSTRAINT: Back-side reflection must not start yet.
    # t1s_ub = "target reflection start, upper bound."  An upper bound on p.
    # tsm_ub = "target reflection start with margin, upper bound."  An upper bound on p.
    ts_ub_list = [[ab_pq, construct_reflected_point(ab_pq, nu, t_pq)] for t_pq in t_pq_list]
    if len(ts_ub_list) >= 1:
        ts_ub = ts_ub_list[0]  # ts_ub_list is sorted with dominant first.
        tsm_ub = g2d.shift_x(ts_ub, -p_margin)
    else:
        ts_ub = None
        tsm_ub = None

    # Start-pass critical altitude point.
    if atvm_lb and tsm_ub:
        # The assessed top visibility and target 1 reflection start margin constraints always dominate.
        sca_pq = g2d.intersect_rays(atvm_lb, tsm_ub)
        # Ensure that the start-pass point is above the minimum safe altitude.
        msa_pq = g2d.intersect_rays(mnsa_ray, tsm_ub)
        if msa_pq[1] > sca_pq[1]:
            sca_pq = msa_pq
    elif atvm_lb:
        # There is no target point, so background occlusion is the only constraint.
        sca_pq = g2d.intersect_rays(atvm_lb, mnsa_ray)
    elif tsm_ub:
        # There is no background heliostat, so there is no bound on starting to avoid background occlusion.
        sca_pq = g2d.intersect_rays(abvm_lb, tsm_ub)
        # Ensure that the start-pass point is above the minimum safe altitude.
        msa_pq = g2d.intersect_rays(mnsa_ray, tsm_ub)
        if msa_pq[1] > sca_pq[1]:
            sca_pq = msa_pq
    else:
        print(
            "ERROR: In single_heliostat_section_analysis(), No-target, no background case not supported.  Use vanity mode."
        )
        assert False

    # Start locus.
    # This is the set of points from which we can start the scan while meeting all the start constraints.
    if tsm_ub:
        dom_ub = tsm_ub  # Target which dominates for start upper bound.
        # Construct a point along dominant ray, away from sca point and in the proper direction.
        dom0_pq = dom_ub[0]
        dom1_pq = dom_ub[1]
        dp_dom = dom1_pq[0] - dom0_pq[0]
        dq_dom = dom1_pq[1] - dom0_pq[1]
        mu = math.atan2(dq_dom, dp_dom)
        length = 10  # Arbitrary positive number.  Unimportant because it will be extended.
        dom2_pq = [sca_pq[0] + (length * np.cos(mu)), sca_pq[1] + (length * np.sin(mu))]
        # Construct the start locus.
        s_locus = [sca_pq, dom2_pq]
    else:
        print("ERROR: In single_heliostat_section_analysis(), No-target start locus case not supported.\n")
        assert False

    # PASS END ANALYSIS

    # CONSTRAINT: Back-side reflection must have reached top of assessed heliostat.
    # t1e_lb = "target 1 reflection end, lower bound."  A lower bound on p.
    te_lb_list = [[at_pq, construct_reflected_point(at_pq, nu, t_pq)] for t_pq in t_pq_list]
    if len(te_lb_list) >= 1:
        te_lb = te_lb_list[0]  # ts_ub_list is sorted with dominant first.
        tem_lb = g2d.shift_x(te_lb, p_margin)
    else:
        te_lb = None
        tem_lb = None

    # CONSTRAINT: Must not pass assesed heliostat plane.
    # pln_ub = "tangent, upper bound."  An upper bound on p.
    pln_p = at_pq[0] + (length * np.cos(tau))
    pln_q = at_pq[1] + (length * np.sin(tau))
    pln_pq = [pln_p, pln_q]
    pln_ub = [at_pq, pln_pq]

    # End-pass critical altitude point.
    if te_lb:
        # This is the dominant target reflection end margin constraint.
        # Its starting point is the assessed heliostat top edge, which is obviously below the minimum safe altitude.
        eca_pq = g2d.intersect_rays(mnsa_ray, te_lb)
    else:
        print("ERROR: In single_heliostat_section_analysis(), No-target end critical altitude case not supported (1).")
        assert False

    # Construct the end locus.
    # Target end reflection always dominates the plane constraint.
    # The target 1, 2, 3,... constraints are nearly collinear, because the heliostat tops are nearly aligned.
    if te_lb:
        dom_lb = te_lb  # Target always dominates for end lower bound.
        # Construct a point along dominant ray, away from sca point and in the proper direction.
        dom0_pq = dom_lb[0]
        dom1_pq = dom_lb[1]
        dp_dom = dom1_pq[0] - dom0_pq[0]
        dq_dom = dom1_pq[1] - dom0_pq[1]
        mu = math.atan2(dq_dom, dp_dom)
        length = 10  # Arbitrary positive number.
        dom2_pq = [eca_pq[0] + (length * np.cos(mu)), eca_pq[1] + (length * np.sin(mu))]
        # Construct the end locus.
        e_locus = [eca_pq, dom2_pq]
    else:
        print("ERROR: In single_heliostat_section_analysis(), No-target, end critical altitude case not supported (2).")
        assert False

    # Critical altitude.
    A = section_context["path_family_A"]
    B = section_context["path_family_B"]
    C_start = -g2d.homogeneous_line_signed_distance_to_xy(sca_pq, [A, B, 0])
    C_end = -g2d.homogeneous_line_signed_distance_to_xy(eca_pq, [A, B, 0])
    C_critical = max(C_start, C_end)

    # Save the constraints.
    # Save before gaze angle analysis, because some gaze angle analysis routines might want to fetch contsraints.
    constraints = {}
    constraints["h_a_idx"] = h_a_idx  # Assessed heliostat index in assess_heliostat_name_list.
    constraints["h_a_name"] = h_a_name  # Assessed heliostat name.
    constraints["h_t_name_list"] = h_t_name_list  # Reflected target heliostat name list.
    constraints["h_b_name"] = h_b_name  # Background heliostat name.
    constraints["at_pq"] = at_pq  # Top corner of assessed heliostat.
    constraints["ab_pq"] = ab_pq  # Bottom corner of assessed heliostat.
    constraints["t_pq_list"] = t_pq_list  # List of reflection target points.  Might include a fictitious point.
    constraints["bb_pq"] = bb_pq  # Bottom corner of background heliostat.
    constraints["nu"] = nu  # Angle from p axis to assessed heliostat surface normal, measured ccw.
    constraints["abv_lb"] = abv_lb  # Assessed bottom visibility, p lower bound.
    constraints["abvm_lb"] = abvm_lb  # Assessed bottom visibility margin, p lower bound.
    constraints["atv_lb"] = atv_lb  # Assessed top visibility, p lower bound.
    constraints["atvm_lb"] = atvm_lb  # Assessed top visibility margin, p lower bound.
    constraints["ts_ub_list"] = ts_ub_list  # Target reflection start list, p upper bound.
    constraints["ts_ub"] = ts_ub  # Target reflection start, p upper bound.
    constraints["tsm_ub"] = tsm_ub  # Target reflection margin, p upper bound.
    constraints["sca_pq"] = sca_pq  # Path start critical altitude point.
    constraints["s_locus"] = s_locus  # Valid pass start points.
    constraints["te_lb_list"] = te_lb_list  # Target reflection end list, p lower bound.
    constraints["te_lb"] = te_lb  # Target reflection end, p lower bound.
    constraints["tem_lb"] = tem_lb  # Target reflection end margin, p lower bound.
    constraints["pln_ub"] = pln_ub  # Mirror plane, p upper bound.
    constraints["eca_pq"] = eca_pq  # Path end critical altitude point.
    constraints["e_locus"] = e_locus  # Valid pass end points.
    constraints["C_start"] = C_start  # Altitude of start critical point.
    constraints["C_end"] = C_end  # Altitude of end critical point.
    constraints["C_critical"] = C_critical  # Critical altitude, considering both start and end.

    # GAZE ANGLE ANALYSIS
    constraints = single_heliostat_gaze_angle_analysis(section_context, ab_pq, at_pq, s_locus, e_locus, constraints)
    constraints = single_heliostat_gaze_angle_selection(section_context, constraints)

    # Return.
    return constraints


def multi_heliostat_gaze_angle_analysis(pass_constraints):
    # Find the worst-case required limits.
    pass_min_etaC_list = []
    pass_max_etaC_list = []
    per_heliostat_constraints = pass_constraints["per_heliostat_constraints"]
    for h_a_name in per_heliostat_constraints.keys():
        h_a_constraints = per_heliostat_constraints[h_a_name]
        h_a_min_etaC_list = h_a_constraints["min_etaC_list"]
        h_a_max_etaC_list = h_a_constraints["max_etaC_list"]
        if len(pass_min_etaC_list) == 0:
            pass_min_etaC_list = h_a_min_etaC_list.copy()
            pass_max_etaC_list = h_a_max_etaC_list.copy()
        else:
            if len(pass_min_etaC_list) != len(h_a_min_etaC_list):
                print("ERROR: In, multi_heliostat_gaze_angle_analysis(), mismatched min_etaC lengths encountered.")
                assert False
            if len(pass_max_etaC_list) != len(h_a_max_etaC_list):
                print("ERROR: In, multi_heliostat_gaze_angle_analysis(), mismatched max_etaC lengths encountered.")
                assert False
            for pass_min_etaC, h_a_min_etaC, idx in zip(
                pass_min_etaC_list, h_a_min_etaC_list, range(0, len(pass_min_etaC_list))
            ):
                if h_a_min_etaC[0] < pass_min_etaC[0]:
                    pass_min_etaC_list[idx] = h_a_min_etaC.copy()
            for pass_max_etaC, h_a_max_etaC, idx in zip(
                pass_max_etaC_list, h_a_max_etaC_list, range(0, len(pass_max_etaC_list))
            ):
                if h_a_max_etaC[0] > pass_max_etaC[0]:
                    pass_max_etaC_list[idx] = h_a_max_etaC.copy()
    # Save result.
    pass_constraints["pass_min_etaC_list"] = pass_min_etaC_list
    pass_constraints["pass_max_etaC_list"] = pass_max_etaC_list
    # Return.
    return pass_constraints


def multi_heliostat_gaze_angle_selection(section_context, pass_constraints):
    # Fetch the camera vertical field of view.
    camera = section_context["camera"]
    # For now, ignore zoom lens focal length selection and just assume widest angle setting.
    camera_vertical_fov = camera.fov_vertical_max

    # Fetch the gaze angle lower and uppper required limits, as a function of C.
    pass_min_etaC_list = pass_constraints["pass_min_etaC_list"]
    pass_max_etaC_list = pass_constraints["pass_max_etaC_list"]
    # Check input.
    if len(pass_min_etaC_list) != len(pass_max_etaC_list):
        print(
            "ERROR: In, multi_heliostat_gaze_angle_selection(), mismatched min_etaC and max_etaC lengths encountered."
        )
        assert False

    # Shift the gaze angle required limits to obtain constraints on the center gaze angle.
    d_eta = camera_vertical_fov / 2.0  # Shift by half the field of view.
    shifted_min_etaC_list = [[etaC[0] + d_eta, etaC[1]] for etaC in pass_min_etaC_list]
    shifted_max_etaC_list = [[etaC[0] - d_eta, etaC[1]] for etaC in pass_max_etaC_list]

    # Construct the feasible gaze angles.
    envelope_min_etaC_list = []
    envelope_max_etaC_list = []
    for shifted_min_etaC, shifted_max_etaC in zip(shifted_min_etaC_list, shifted_max_etaC_list):
        if shifted_min_etaC[0] >= shifted_max_etaC[0]:
            envelope_min_etaC_list.append(shifted_max_etaC)  # Lines crossed, so we swap min.max sense.
            envelope_max_etaC_list.append(shifted_min_etaC)  #

    # Shrink by gaze tolerance.
    gaze_tolerance = section_context["gaze_tolerance"]
    shrunk_min_etaC_list = []
    shrunk_max_etaC_list = []
    for envelope_min_etaC, envelope_max_etaC in zip(envelope_min_etaC_list, envelope_max_etaC_list):
        C = envelope_min_etaC[1]
        envelope_min_eta = envelope_min_etaC[0]
        envelope_max_eta = envelope_max_etaC[0]
        shrunk_min_eta = envelope_min_eta + gaze_tolerance
        shrunk_max_eta = envelope_max_eta - gaze_tolerance
        if shrunk_min_eta <= shrunk_max_eta:
            shrunk_min_etaC_list.append([shrunk_min_eta, C])
            shrunk_max_etaC_list.append([shrunk_max_eta, C])

    # Clip to eta limits.
    eta_min = section_context["eta_min"]
    eta_max = section_context["eta_max"]
    clipped_min_etaC_list = []
    clipped_max_etaC_list = []
    for envelope_min_etaC, envelope_max_etaC in zip(shrunk_min_etaC_list, shrunk_max_etaC_list):
        C = envelope_min_etaC[1]
        envelope_min_eta = envelope_min_etaC[0]
        envelope_max_eta = envelope_max_etaC[0]
        clipped_min_eta = max(eta_min, envelope_min_eta)
        clipped_max_eta = min(eta_max, envelope_max_eta)
        if clipped_min_eta <= clipped_max_eta:
            clipped_min_etaC_list.append([clipped_min_eta, C])
            clipped_max_etaC_list.append([clipped_max_eta, C])

    # Fork based on desired gaze control.
    gaze_type = section_context["gaze_type"]
    if gaze_type == "constant":
        # Select gaze angle.
        if len(clipped_min_etaC_list) > 0:
            lowest_min_etaC = clipped_min_etaC_list[0]
            lowest_max_etaC = clipped_max_etaC_list[0]
            selected_cacg_eta = (lowest_min_etaC[0] + lowest_max_etaC[0]) / 2.0
            # Select altitude.
            C_clipped = lowest_min_etaC[1]
            max_C_critical = -math.inf
            per_heliostat_constraints = pass_constraints["per_heliostat_constraints"]
            for h_a_name in per_heliostat_constraints.keys():
                h_a_constraints = per_heliostat_constraints[h_a_name]
                h_a_C_critical = h_a_constraints["C_critical"]
                if h_a_C_critical > max_C_critical:
                    max_C_critical = h_a_C_critical
            selected_cacg_C = max(C_clipped, max_C_critical)
        else:
            print("WARNING: In multi_heliostat_gaze_angle_selection(), infeasible case encountered.")
            print("               len(envelope_min_etaC_list) = ", len(envelope_min_etaC_list))
            print("               len(shrunk_min_etaC_list)   = ", len(shrunk_min_etaC_list))
            print("               len(clipped_min_etaC_list)  = ", len(clipped_min_etaC_list))
            # Set values that will force the following code to clip the altitude and gaze angle.
            selected_cacg_eta = np.deg2rad(-90.0)  # Extreme value.
            selected_cacg_C = 90.0  # m.   Signal value.

        # If result exceeds the maximum altitude, then clip to that alttiude, while preseving maximum eta constraint.
        #
        # This is because the maximum eta constraint ensures that we will see the heliostat first without back-side
        # feature reflection, which provides two benefits:
        #    1. As features sweep up the mirror, we will obtain optical targets all the way from the bottom up.
        #    2. The mirror's first appearance will generally be reflecting sky only, facilitating automatic recognition.
        #    3. We will see the full heliostat reflecting sky, supporting per-facet inspection tasks across all mirrors.
        #
        max_C = section_context["maximum_altitude"]
        if selected_cacg_C > max_C:
            print("NOTE: In multi_heliostat_gaze_angle_selection(), clipping to max altitude.")
            # Pick altitude, then lookup eta that ensures mirror will be seen intially facing all sky.
            if len(shifted_max_etaC_list) < 2:
                print(
                    "ERROR: In multi_heliostat_gaze_angle_selection(), unexpected short shifted_max_etaC_list encountered."
                )
                assert False
            selected_cacg_C = max_C  # m.
            if shifted_max_etaC_list[0][1] > selected_cacg_C:
                print("ERROR: In multi_heliostat_gaze_angle_selection(), shifted_max_etaC_list starts above C.")
                assert False
            # Search for point on eta = f(C) curve.
            selected_cacg_eta = math.inf
            prev_C = -math.inf
            for max_etaC in shifted_max_etaC_list:
                C = max_etaC[1]
                if C >= selected_cacg_C:
                    selected_cacg_eta = max_etaC[0]
                    break
                prev_C = C
            if selected_cacg_eta == math.inf:
                print("ERROR: In multi_heliostat_gaze_angle_selection(), unexpected selected_cacg_eta == math.inf.")
                assert False
            if prev_C > selected_cacg_C:
                print("ERROR: In multi_heliostat_gaze_angle_selection(), unexpected prev_C > selected_cacg_C.")
                assert False
            if C < selected_cacg_C:
                print("ERROR: In multi_heliostat_gaze_angle_selection(), unexpected C < selected_cacg_C.")
                assert False
            # Shrink.
            # Recall max and min swapped.
            selected_cacg_eta += gaze_tolerance
            # Clip.
            if selected_cacg_eta < eta_min:
                selected_cacg_eta = eta_min
    else:
        print(
            'ERROR: In multi_heliostat_gaze_angle_selection(), unexpected gaze_type="'
            + str(gaze_type)
            + '" encountered.'
        )
        assert False

    # Assemble selected (eta, C) pair.
    selected_cacg_etaC = [selected_cacg_eta, selected_cacg_C]

    # Save result.
    pass_constraints["shifted_min_etaC_list"] = shifted_min_etaC_list
    pass_constraints["shifted_max_etaC_list"] = shifted_max_etaC_list
    pass_constraints["envelope_min_etaC_list"] = envelope_min_etaC_list
    pass_constraints["envelope_max_etaC_list"] = envelope_max_etaC_list
    pass_constraints["clipped_min_etaC_list"] = clipped_min_etaC_list
    pass_constraints["clipped_max_etaC_list"] = clipped_max_etaC_list
    pass_constraints["shrunk_min_etaC_list"] = shrunk_min_etaC_list
    pass_constraints["shrunk_max_etaC_list"] = shrunk_max_etaC_list
    pass_constraints["selected_cacg_etaC"] = selected_cacg_etaC  # "cacg" == "constant altitude, constant gaze"

    # Return.
    return pass_constraints


def multi_heliostat_construct_flight_path(section_context, assess_heliostat_name_list, pass_constraints):
    # Fetch the selected gaze angle and altitude.
    selected_cacg_etaC = pass_constraints["selected_cacg_etaC"]  # "cacg" == "constant altitude, constant gaze"

    # Fetch family of flight paths.
    A = section_context["path_family_A"]
    B = section_context["path_family_B"]

    # Construct selected flight path line.
    selected_cacg_A = A
    selected_cacg_B = B
    selected_cacg_C = selected_cacg_etaC[1]
    selected_cacg_line = [selected_cacg_A, selected_cacg_B, selected_cacg_C]

    # Construct start scan point.
    first_heliostat_name = assess_heliostat_name_list[0]
    per_heliostat_constraints = pass_constraints["per_heliostat_constraints"]
    first_constraints = per_heliostat_constraints[first_heliostat_name]
    first_s_locus = first_constraints["s_locus"]
    selected_cacg_start_pq = g2d.intersect_lines(
        selected_cacg_line, g2d.homogeneous_line(first_s_locus[0], first_s_locus[1])
    )

    # Construct end scan point.
    last_heliostat_name = assess_heliostat_name_list[-1]
    per_heliostat_constraints = pass_constraints["per_heliostat_constraints"]
    last_constraints = per_heliostat_constraints[last_heliostat_name]
    last_e_locus = last_constraints["e_locus"]
    selected_cacg_end_pq = g2d.intersect_lines(
        selected_cacg_line, g2d.homogeneous_line(last_e_locus[0], last_e_locus[1])
    )

    # Construct flight selected path segment.
    selected_cacg_segment = [selected_cacg_start_pq, selected_cacg_end_pq]

    # Save result.
    pass_constraints["selected_cacg_line"] = selected_cacg_line
    pass_constraints["selected_cacg_segment"] = selected_cacg_segment

    # Return.
    return pass_constraints


def multi_heliostat_vertical_fov_analysis(pass_constraints):
    # Fetch the gaze angle lower and uppper required limits, as a function of C.
    pass_min_etaC_list = pass_constraints["pass_min_etaC_list"]
    pass_max_etaC_list = pass_constraints["pass_max_etaC_list"]
    # Check input.
    if len(pass_min_etaC_list) != len(pass_max_etaC_list):
        print(
            "ERROR: In, multi_heliostat_vertical_fov_analysis(), mismatched min_etaC and max_etaC lengths encountered."
        )
        assert False
    # Assemble required field of view list.
    vertical_fovC_list = []
    for pass_min_etaC, pass_max_etaC in zip(pass_min_etaC_list, pass_max_etaC_list):
        vertical_fov = pass_max_etaC[0] - pass_min_etaC[0]
        vertical_fovC_list.append([vertical_fov, pass_min_etaC[1]])
    # Save result.
    pass_constraints["vertical_fovC_list"] = vertical_fovC_list
    # Return.
    return pass_constraints


def assemble_single_heliostat_etaC_dict(pass_constraints):
    # Collect selected (eta,C) results for each heliostat.
    selected_cacg_etaC_dict = {}
    per_heliostat_constraints = pass_constraints["per_heliostat_constraints"]
    for h_a_name in per_heliostat_constraints.keys():
        h_a_constraints = per_heliostat_constraints[h_a_name]
        selected_cacg_etaC = h_a_constraints["selected_cacg_etaC"]  # "cacg" == "constant altitude, constant gaze"
        selected_cacg_etaC_dict[h_a_name] = selected_cacg_etaC
    # Save result.
    pass_constraints["selected_cacg_etaC_dict"] = selected_cacg_etaC_dict
    # Return.
    return pass_constraints


def section_analysis(section_context, heliostat_name_list, assess_heliostat_name_list):
    # Notify progress.
    print("Constructing UFACET section " + heliostat_name_list[0] + "-" + heliostat_name_list[-1] + " analysis...")

    # Construct minimum safe altitude line for this section.
    section_context = construct_minimum_safe_altitude_line(section_context, heliostat_name_list)

    # Construct family of flight paths, indexed by altitude.
    section_context = construct_flight_path_family(section_context)

    # Per-heliostat constraint analysis.
    per_heliostat_constraints = {}
    for assess_heliostat_name in assess_heliostat_name_list:
        constraints = single_heliostat_section_analysis(section_context, heliostat_name_list, assess_heliostat_name)
        per_heliostat_constraints[assess_heliostat_name] = constraints

    # Full pass analysis.
    pass_constraints = {}
    pass_constraints["per_heliostat_constraints"] = per_heliostat_constraints
    pass_constraints = multi_heliostat_gaze_angle_analysis(pass_constraints)
    pass_constraints = multi_heliostat_gaze_angle_selection(section_context, pass_constraints)
    pass_constraints = multi_heliostat_vertical_fov_analysis(pass_constraints)
    pass_constraints = multi_heliostat_construct_flight_path(
        section_context, assess_heliostat_name_list, pass_constraints
    )

    # Sequence of single-heliostat analyses.
    pass_constraints = assemble_single_heliostat_etaC_dict(pass_constraints)

    # Return.
    return pass_constraints
