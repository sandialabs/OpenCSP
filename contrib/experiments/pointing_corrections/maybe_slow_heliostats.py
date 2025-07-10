import datetime as dt
import re

from contrib.site_specific.NSTTFHeliostatLogsParser import NSTTFHeliostatLogsParser
from opencsp import opencsp_settings
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

if __name__ == "__main__":
    experiment_path = ft.join(
        opencsp_settings["opencsp_root_path"]["collaborative_dir"],
        "NSTTF_Optics/Experiments/2024-06-16_FluxMeasurement",
    )
    log_path = ft.join(experiment_path, "2_Data/context/heliostat_logs")
    log_name_exts = ft.files_in_directory(log_path)
    log_path_name_exts = [ft.join(log_path, log_name_ext) for log_name_ext in log_name_exts]
    # example log name: "log_ 5_ 3_2024_13" for May 3rd 2024 at 1pm
    # replacement: "2024/5/3"
    save_path = ft.join(experiment_path, "4_Analysis/maybe_slow_13_more_time")
    from_regex = re.compile(r".*_ ?([0-9]{1,2})_ ?([0-9]{1,2})_([0-9]{4})_ ?([0-9]{1,2})")
    to_pattern = r"\3/\1/\2"

    rows = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    ncols = [9, 9, 9, 10, 11, 12, 14, 14, 14, 6]
    expected_hnames = []
    for row, ncols in zip(rows, ncols):
        for col in range(1, ncols + 1):
            for ew in ["E", "W"]:
                expected_hnames.append(f"_{row:02d}{ew}{col:02d}")

    times: dict[tuple[dt.datetime, dt.datetime]] = {
        "_05W01": (dt.time(13, 18, 34), dt.time(13, 18, 54)),
        "_05E06": (dt.time(13, 19, 31), dt.time(13, 19, 52)),
        "_09W01": (dt.time(13, 20, 29), dt.time(13, 20, 49)),
        "_14W01": (dt.time(13, 21, 21), dt.time(13, 21, 44)),
        "_14W06": (dt.time(13, 22, 31), dt.time(13, 22, 56)),
    }
    to_datetime = lambda time: dt.datetime.combine(dt.date(2024, 6, 16), time)
    for hel in times:
        enter, leave = times[hel]
        times[hel] = to_datetime(enter), to_datetime(leave)
    total_delta = times["_14W06"][1] - times["_05W01"][0]
    lt.info(f"{total_delta=}")

    parser = NSTTFHeliostatLogsParser.NsttfLogsParser()
    parser.filename_datetime_replacement = (from_regex, to_pattern)
    parser.filename_datetime_format = r"%Y/%m/%d"
    parser.load_heliostat_logs(
        log_path_name_exts, usecols=["Helio", "Time ", "X Targ", "Z Targ", "Az", "Az Targ", "Elev", "El Targ"]
    )
    parser.check_for_missing_heliostats(expected_hnames)

    maybe_slow_helios = [
        "_06W03",
        "_6E08",
        "_08E08",
        "_08W08",
        "_10E10",
        "_10W08",
        "_11E11",
        "_11E07",
        "_11W09",
        "_12W05",
        "_12E10",
        "_13E09",
        "_14E06",
    ]
    series_columns_labels_list = [
        {'Az': "{helio}Az", 'Az Targ': "{helio}AzTarg"},
        {'Elev': "{helio}El", 'El Targ': "{helio}ElTarg"},
    ]

    for heliostat in maybe_slow_helios:
        helio = heliostat.lstrip("_")

        for series_columns_labels in series_columns_labels_list[1:]:
            title = helio + " " + ",".join([s for s in series_columns_labels])
            for series in series_columns_labels:
                series_columns_labels[series] = series_columns_labels[series].replace("{helio}", helio)

            fig_record = parser.prepare_figure(title, "Time", "X Targ (m)")
            hparser = parser.filter(
                heliostat_names=heliostat, datetime_range=(dt.time(13, 26, 50), dt.time(13, 29, 30))
            )
            hparser.plot('Time ', series_columns_labels, scatter_plot=False)

            # # datetimes = hparser.datetimes
            # # if len(datetimes) > 4:
            # #     xticks = []
            # #     for i in range(0, len(datetimes), int(len(datetimes) / 4)):
            # #         datetime: pandas.DatetimeIndex = datetimes[i]
            # #         xticks.append((datetime, f"{datetime.time}"))
            # #     fig_record.view.axis.set_xticks([tick_label[0] for tick_label in xticks], [
            # #                                     tick_label[1] for tick_label in xticks])
            # xticks = [dt.datetime(2024, 6, 16, 13, 26, 50) + dt.timedelta(s) for s in range(0, 80, 20)]
            # xlabels = [str(xtick) for xtick in xticks]
            # fig_record.view.axis.set_xticks(xticks, xlabels)

            fig_record.view.show(legend=True, block=False, grid=True)
            fig_record.save(save_path, title, "png")
