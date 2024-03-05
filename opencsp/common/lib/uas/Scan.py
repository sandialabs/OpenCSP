"""


"""

import opencsp.common.lib.uas.ScanPass as sp


class Scan():
    """
    A flight scan, comprised of a sequence of scan passes.
    """

    def __init__(self,
                 scan_passes, # List of ScanPass objects.
                 ):

        super(Scan, self).__init__()

        # Check input.
        if len(scan_passes) == 0:
            print('In Scan constructor, empty list of scan passes encountered.')
            assert False

        # Input parameters.
        self.passes = scan_passes


    # ACCESS

    def waypoints(self):
        waypoint_list = []
        for scan_pass in self.passes:
            pass_waypoint_list = scan_pass.waypoints()
            waypoint_list = waypoint_list + pass_waypoint_list
        return waypoint_list


    def locale(self):
        return self.passes[0].locale()


    # MODIFICATION
    
    def set_scan_pass_numbers(self, next_scan_index):
        for scan_pass in self.passes:
            scan_pass.idx = next_scan_index
            next_scan_index += 1
        return next_scan_index


    # RENDERING
    
    def draw(self, view, scan_pass_styles):
        for scan_pass in self.passes:
            scan_pass.draw(view, scan_pass_styles)


# -------------------------------------------------------------------------------------------------------
# CONSTRUCTION FUNCTIONS
#

def construct_scan_given_segments_of_interest(list_of_xyz_segments, scan_parameters):
    # Notify progress.
    print('Constructing scan given segments of interest...')

    # We assume the input segment list is parallel in the sense that each segment points 
    # from p0 --> p1 in the same direction.
    # However, to maximize scanning that can be done within a limited UAS range, we will 
    # fly a back-and-forth boustrophedon pattern,* so we reverse every other segment.
    # * Thanks to Laura McNamara for this excellent word!
    #
    list_of_xyz_segments_2 = []
    idx = 0
    for segment_xyz in list_of_xyz_segments:
        if (idx % 2) == 0:
            list_of_xyz_segments_2.append(segment_xyz)
        else:
            reverse_segment_xyz = [segment_xyz[1], segment_xyz[0]]
            list_of_xyz_segments_2.append(reverse_segment_xyz)
        idx += 1
    # Construct scan passes.
    passes = []
    idx = 0
    for segment_xyz in list_of_xyz_segments_2:
        if (not scan_parameters['fly_forward_backward']) or ((idx % 2) == 0):
            fly_backward = False
        else:
            fly_backward = True
        passes.append(sp.construct_scan_pass_given_segment_of_interest(segment_xyz,
                                                                       fly_backward,
                                                                       scan_parameters))
        idx += 1
    # Construct scan.
    scan = Scan(passes)
    # Return.
    return scan


def construct_scan_given_UFACET_scan_passes(ufacet_scan_pass_list, scan_parameters):
    # Notify progress.
    print('Constructing scan given UFACET scan passes...')

    # UFACET scans are always specified by line segments that are "parallel" in the sense 
    # that each segment points from p0 --> p1 in roughly the same direction.
    # However, to maximize scanning that can be done within a limited UAS range, we will 
    # fly a back-and-forth boustrophedon pattern,* so we reverse every other segment.
    # * Thanks to Laura McNamara for this excellent word!
    #
    # Because UFACET requires teh camera to face the mirrors during each pass, the 
    # UAS flies backward for every other pass.
    #
    # Construct scan passes.
    passes = []
    idx = 0
    for ufacet_scan_pass in ufacet_scan_pass_list:
        if (idx % 2) == 0:
            fly_backward = False
        else:
            fly_backward = True
        passes.append(sp.construct_scan_pass_given_UFACET_scan_pass(ufacet_scan_pass,
                                                                    fly_backward,
                                                                    scan_parameters))
        idx += 1
    # Construct scan.
    scan = Scan(passes)
    # Return.
    return scan
