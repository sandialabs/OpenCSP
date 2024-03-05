# Open CSP directory app/ufacet-s/

Code under this directory is not actively maintained by the OpenCSP team.

This is experimental code, still in early stages of exploration and development.

# ufacet-s

High-speed scanning of a heliostat field with a flying drone, to assess heliostat state of health and measure optical error.

The computation proceeds in several stages, listed below in sequential order matching the prefix numbers.

# 1000_plan_flight

Given a heliostat field, its configuration, and a specifcation of the heliostats to measure, generate a drone flight plan to scan the selected heliostats.

# 2000_sync_flight_data

After a flight video has been collected, identify the relative time between various sensor streams, such as a flight log, GPS log, and video.

# 3000_find_key_heliostat

Given a flight video synchronized with its flight log, identify key frames conataining heliostats of interest, and then find the heliostat features within that frame.

# 4000_follow_heliostat

Given a flight video and key heliostats found with their features, track those features forward and backward in time.

# 5000_3d_reconstruction

Given observed heliostat feature tracks, construct a best-fit 3-d representation of each observed heliostat.

# 6000_track_reflection

Search the video for features reflected in heliostat, and track them over time.

# 7000_surface_normal_map

Based on above information, estimate the mirror surface normal at various measurement points on the mirror.

# 8000_output_analysis

Given a surface nomral map and other measured aprameters, output analysis plots and reports assessing mirror state of health and optical accuracy.
