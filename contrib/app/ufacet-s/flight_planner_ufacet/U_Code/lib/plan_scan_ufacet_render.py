"""
Rendering the construction of a UFACET-s scan.



"""


def draw_ufacet_scan(figure_control, scan, render_control_scan_section_analysis):
    # Render the analysis.
    for scan_pass in scan.passes:
        scan_pass.ufacet_scan_pass().draw_section_analysis(figure_control, render_control_scan_section_analysis)
