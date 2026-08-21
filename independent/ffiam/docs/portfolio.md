# FFIAM Portfolio

FFIAM (Full-Field Irradiance Analysis Model) is a parameterized volumetric analysis library that assesses and summarizes the expected irradiance of a heliostat field in standby. It evaluates the irradiance profile of the entire field volume to find hotspot locations and flux impacts on avian paths.

FFIAM is a C++ library, run from the command line or integrated into a workflow. Two integrations ship today: Visualization and Data. Each heliostat is a grid of facets with a focal point. Heliostat design and field layout are set by parameters, CSV import, or JSON config. Standby aim strategies include single-point, ring, split-ring, and constant-vector.

Figures come from both flows. Sources are in [`docs/images/`](images/).

## Visualization Workflow

A custom 3D field renderer built with Unreal Engine. It renders the input CSP field and the expected flux voxels in real time, navigable by the user.

| View                                         | Figure                                                               |
|----------------------------------------------|----------------------------------------------------------------------|
| NSTTF field, standby flux above the receiver | ![NSTTF standby flux](images/ffiam-nsttf-flux.jpg)                   |
| Radial field, point standby (top-down)       | ![Radial field, point standby](images/ffiam-field-point-topdown.jpg) |

### Aim strategies
The model includes several aim strategies, and accepts per-heliostat CSV files for custom configurations.

| Strategy | Description | Figure |
|----------|-------------|--------|
| Single-point | All heliostats aim at one location; flux converges into a concentrated column | ![Point standby](images/ffiam-aim-point.jpg) |
| Ring | Aim distributed around an annulus above the tower; flux forms a broader elevated region | ![Ring standby](images/ffiam-aim-ring.jpg) |
| Split-ring | Each azimuthal slice fans across the annulus, redistributing flux around the receiver | ![Split-ring standby](images/ffiam-aim-splitring.jpg) |
| Constant-vector | All heliostats aim in a fixed direction (e.g. stow); flux spreads across the field | ![Vector standby](images/ffiam-aim-vector.jpg) |

### Voxel resolution

The airspace is partitioned into voxels. Default resolution is 2 m/side, an 8x improvement in per-voxel volume over the original 4 m.
Depending on memory constraints, smaller sites can be set to 1 m voxel resolution.

| View                            | Figure |
|---------------------------------|--------|
| 4 m voxels (opaque for clarity) | ![4 m voxels](images/ffiam-ue-voxels-4m.png) |
| 2 m voxels                      | ![2 m voxels](images/ffiam-ue-voxels-2m.png) |
| 1 m voxels, flux detail         | ![1 m voxel detail](images/ffiam-voxels-detail.jpg) |

## Data flow

A Python module (`pyffiam`) that runs streamlined analyses and generates heatmap and profile charts plus tabulated summary tables, suitable for reports and presentations. It wraps the same C++/CUDA library as the Visualization flow.

### Volumetric voxel irradiance

Irradiance is resolved through the 3D volume and viewable in any cardinal plane (east-north, north-up, east-up).

| View | Figure |
|------|--------|
| NSTTF, plan (east-north) | ![NSTTF east-north](images/sample-ffiam-nsttf-en.png) |
| NSTTF, elevation (north-up) | ![NSTTF north-up](images/sample-ffiam-nsttf-nu.png) |
| NSTTF, hotspot zoom | ![NSTTF east-north zoom](images/sample-ffiam-nsttf-en-zoom.png) |

### Time-resolved animation

Fields are computed across solar position and rendered as animations. The high-irradiance region moves as the sun moves.

| View | Figure |
|------|--------|
| NSTTF, east-north | ![NSTTF animation](images/sample-ffiam-nsttf-en-anim-white.gif) |
| NSTTF, east-up | ![NSTTF east-up animation](images/sample-ffiam-nsttf-eu-anim.gif) |
| SampleV1, east-north | ![SampleV1 animation](images/sample-ffiam-v1-en-anim.gif) |

### Flight-path irradiance and radiant exposure

FFIAM evaluates irradiance along a polyline path. The user gives (east, north, up) vertices and an average velocity per path. It reports instantaneous irradiance and cumulative radiant exposure (dose) over time, for airspace-safety work.

| View | Figure |
|------|--------|
| Irradiance along a path | ![Path irradiance](images/sample-path-irradiance.png) |
| Radiant exposure (dose) along a path | ![Path radiant exposure](images/sample-path-radiant-exposure.png) |

### Site presets

The pipeline runs the real NSTTF field, synthetic SampleV1/V2/V3 sites, and parametric Radial fields. Users can reproduce these or load their own field via CSV/JSON.

| View | Figure |
|------|--------|
| SampleV1, east-up | ![SampleV1 east-up](images/sample-ffiam-v1-eu.png) |
| SampleV1, hotspot zoom | ![SampleV1 east-up zoom](images/sample-ffiam-v1-eu-zoom.png) |

## Scale

V3.0 handles a 1.6 km field radius, 300 m airspace height, over 11,000 heliostats, and 300 million voxels per analysis at 2 m resolution, with 5 rays per heliostat facet.

## Reproducing these results

Data-flow figures are generated by examples in [`pyffiam/src/pyffiam/examples.py`](../pyffiam/src/pyffiam/examples.py):

- `assess_nsttf()`: real NSTTF field (volumetric, animation)
- `assess_radial_small()`: parametric radial field, runnable baseline (presets)
- `assess_sample_v2()`: synthetic site (presets)
- `assess_radial()`: full-size radial field (1.6 km, 10,000 heliostats)

See [`README.md`](../README.md) for install and usage, and [`PYFFIAM-GUIDE.md`](../PYFFIAM-GUIDE.md) for the Python API.
