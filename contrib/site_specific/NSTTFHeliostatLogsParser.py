import copy
import datetime as dt
import re

import numpy as np
import pandas
import pandas.core

import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.typing_tools as tt


class NSTTFHeliostatLogsParser:
    """Parser for NSTTF style log output from the heliostat control software."""

    def __init__(
        self, name: str, dtype: dict[str, any], date_column_formats: dict[str, str], heliostat_name_column: str
    ):
        # register inputs
        self.name = name
        self.dtype = dtype
        self.date_column_formats = date_column_formats
        self.heliostat_name_column = heliostat_name_column

        # internal values
        self.filename_datetime_replacement: tuple[re.Pattern, str] = None
        self.filename_datetime_format: str = None

        # plotting values
        self.figure_rec: rcfr.RenderControlFigureRecord = None
        self.nplots = 0
        self.parent_parser: NSTTFHeliostatLogsParser = None

    @classmethod
    def NsttfLogsParser(cls):
        dtype = {
            # "Main T": ,
            "Time": str,
            "Helio": str,
            "Mode": str,
            "Sleep": str,
            "Track": int,
            "X Targ": float,
            "Y Targ": float,
            "Z Targ": float,
            "az offset": float,
            "el offset": float,
            # "reserved": ,
            "Az Targ": float,
            "El Targ": float,
            "Az": float,
            "Elev": float,
            # "Az Amp": ,
            # "El Amp": ,
            # "Az Falt": ,
            # "El Falt": ,
            # "Az Cnt": ,
            # "El Cnt": ,
            # "Az Drive": ,
            # "El Drive": ,
            "Trk Time": float,
            "Exec Time": float,
            "Delta Time": float,
            # "Ephem Num": ,
            # "Status Word": ,
        }
        # date format for "09:59:59.999" style timestamp
        # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
        date_column_formats = {"Time ": r"%H:%M:%S.%f"}
        heliostat_name_column = "Helio"

        return cls('NsttfLogsParser', dtype, date_column_formats, heliostat_name_column)

    @property
    def column_names(self) -> list[str]:
        return list(self.data.columns)

    @property
    def heliostats(self) -> tt.Series[str]:
        return self.column(self.heliostat_name_column)

    @property
    def datetimes(self) -> tt.Series[dt.datetime]:
        dt_col = next(iter(self.date_column_formats.keys()))
        return self.column(dt_col)

    @datetimes.setter
    def datetimes(self, datetimes: pandas.Series):
        dt_col = next(iter(self.date_column_formats.keys()))
        self.data[dt_col] = datetimes

    def column(self, column_name: str) -> pandas.Series:
        if column_name is None:
            lt.error_and_raise(ValueError, "Error in HeliostatLogsParser.column(): column_name is None")
        if column_name not in self.column_names:
            lt.error_and_raise(
                KeyError,
                "Error in HeliostatLogsParser.column(): "
                + f"can't find column \"{column_name}\", should be one of {self.column_names}",
            )
        return self.data[column_name]

    def load_heliostat_logs(self, log_path_name_exts: str | list[str], usecols: list[str] = None, nrows: int = None):
        # normalize input
        if isinstance(log_path_name_exts, str):
            log_path_name_exts = [log_path_name_exts]
        for i, log_path_name_ext in enumerate(log_path_name_exts):
            log_path_name_exts[i] = ft.norm_path(log_path_name_ext)

        # validate input
        for log_path_name_ext in log_path_name_exts:
            if not ft.file_exists(log_path_name_ext):
                lt.error_and_raise(
                    FileNotFoundError,
                    f"Error in HeliostatLogsParser({self.name}).load_heliostat_logs(): "
                    + f"file \"{log_path_name_ext}\" does not exist!",
                )

        # load the logs
        data_list: list[pandas.DataFrame] = []
        for i, log_path_name_ext in enumerate(log_path_name_exts):
            lt.info(f"Loading {log_path_name_ext}... ")

            data = pandas.read_csv(
                log_path_name_ext,
                delimiter="\t",
                header='infer',
                # parse_dates=self.parse_dates,
                dtype=self.dtype,
                skipinitialspace=True,
                # date_format=self.date_format,
                usecols=usecols,
                nrows=nrows,
            )
            data_list.append(data)
        self.data = pandas.concat(data_list)
        data_list.clear()

        # try to guess the date from the file name
        date = None
        if self.filename_datetime_format is not None:
            _, log_name, _ = ft.path_components(log_path_name_ext)
            if self.filename_datetime_replacement is not None:
                repl_pattern, repl_sub = self.filename_datetime_replacement
                formatted_log_name: str = repl_pattern.sub(repl_sub, log_name)
                date = formatted_log_name
            else:
                date = log_name

        # parse any necessary dates
        # masterlog _ 5_ 3_2024_13.lvm
        for date_col in self.date_column_formats:
            dt_format = self.date_column_formats[date_col]
            col_to_parse = self.data[date_col]
            if not r"%d" in dt_format and r"%j" not in dt_format:
                if date is not None:
                    col_to_parse = date + " " + self.data[date_col]
                    dt_format = self.filename_datetime_format + " " + dt_format

            self.data[date_col] = pandas.to_datetime(col_to_parse, format=dt_format)

        lt.info("..done")

    def filter(
        self,
        heliostat_names: str | list[str] = None,
        columns_equal: list[tuple[str, any]] = None,
        columns_almost_equal: list[tuple[str, float]] = None,
        datetime_range: tuple[dt.datetime, dt.datetime] | tuple[dt.time, dt.time] = None,
    ) -> "NSTTFHeliostatLogsParser":
        if isinstance(heliostat_names, str):
            heliostat_names = [heliostat_names]

        # copy of the data to be filtered
        new_data = self.data
        if heliostat_names is not None:
            new_data = new_data[new_data[self.heliostat_name_column].isin(heliostat_names)]

        # filter by datetime
        if datetime_range is not None:
            dt_col = next(iter(self.date_column_formats.keys()))
            if isinstance(datetime_range[0], dt.datetime):
                # user specified dates+times
                matches = (new_data[dt_col] >= datetime_range[0]) & (new_data[dt_col] < datetime_range[1])
            elif isinstance(datetime_range[0], dt.time):
                # user specified just times, select by all matches across all dates
                dates: set[dt.date] = set([val.date() for val in self.datetimes])
                matches = np.full_like(new_data[dt_col], fill_value=False, dtype=np.bool_)
                for date in dates:
                    fromval = pandas.to_datetime(dt.datetime.combine(date, datetime_range[0]))
                    toval = pandas.to_datetime(dt.datetime.combine(date, datetime_range[1]))
                    matches |= (new_data[dt_col] >= fromval) & (new_data[dt_col] < toval)
            else:
                lt.error_and_raise(
                    ValueError,
                    "Error in HeliostatLogsParser.filter(): "
                    + f"unexpected type for datetime_range, expected datetime or time but got {type(datetime_range[0])}",
                )
            new_data = new_data[matches]

        # filter by generic exact values
        if columns_equal is not None:
            for column_name, value in columns_almost_equal:
                new_data = new_data[new_data[column_name] == value]

        # filter by generic approximate values
        if columns_almost_equal is not None:
            # definition for 'almost_equal' from np.testing.assert_almost_equal()
            # abs(desired-actual) < float64(1.5 * 10**(-decimal))
            decimal = 7
            error_bar = 1.5 * 10 ** (-decimal)
            for column_name, value in columns_almost_equal:
                matches = np.abs(new_data[column_name] - value) < error_bar
                new_data = new_data[matches]

        # create a copy with the filtered data
        ret = copy.copy(self)
        ret.data = new_data
        ret.parent_parser = self

        return ret

    def check_for_missing_heliostats(self, expected_heliostat_names: list[str]) -> tuple[list[str], list[str]]:
        extra_hnames, missing_hnames = [], copy.copy(expected_heliostat_names)
        hnames = set(self.data["Helio"])
        for hname in hnames:
            if hname in missing_hnames:
                missing_hnames.remove(hname)
            else:
                extra_hnames.append(hname)

        lt.info(f"Missing {len(missing_hnames)} expected heliostats: {missing_hnames}")
        lt.info(f"Found {len(extra_hnames)} extra heliostats: {extra_hnames}")

        return missing_hnames, extra_hnames

    def prepare_figure(self, title: str = None, x_label: str = None, y_label: str = None):
        # normalize input
        if title is None:
            title = f"{self.__class__.__name__} ({self.name})"
        if x_label is None:
            x_label = "x"
        if y_label is None:
            y_label = "y"

        # get the plot ready
        view_spec = vs.view_spec_pq()
        axis_control = rca.RenderControlAxis(x_label=x_label, y_label=y_label)
        figure_control = rcf.RenderControlFigure(tile=False)
        self.fig_record = fm.setup_figure(
            figure_control=figure_control,
            axis_control=axis_control,
            view_spec=view_spec,
            equal=False,
            number_in_name=False,
            title=title,
            code_tag=f"{__file__}.build_plot()",
        )
        self.nplots = 0

        return self.fig_record

    def plot(self, x_axis_column: str, series_columns_labels: dict[str, str], scatter_plot=False):
        view = self.fig_record.view
        x_values = self.data[x_axis_column].to_list()

        # populate the plot
        for series_column in series_columns_labels:
            series_label = series_columns_labels[series_column]
            series_values = self.data[series_column].to_list()
            if scatter_plot:
                view.draw_pq(
                    (x_values, series_values),
                    label=series_label,
                    style=rcps.default(color=color._PlotColors()[self.nplots]),
                )
            else:
                view.draw_pq_list(
                    list(zip(x_values, series_values)),
                    label=series_label,
                    style=rcps.outline(color=color._PlotColors()[self.nplots]),
                )
            self.nplots += 1

        # bubble up the nplots value to the parent parser
        curr_parser = self
        while curr_parser.parent_parser is not None:
            curr_parser.parent_parser.nplots = curr_parser.nplots
            curr_parser = curr_parser.parent_parser
