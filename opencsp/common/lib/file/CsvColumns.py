from collections import namedtuple
import re

import opencsp.common.lib.tool.log_tools as lt

_ColumnHeader = namedtuple("ColumnHeader", ["name", "aliases", "idx"])


class CsvColumns:
    """
    A class to help parse CSV files with a tentative structure by finding column name matches.

    This class allows for the definition of expected column names and their aliases,
    and provides methods to parse the header and data rows of a CSV file.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __init__(self, columns: dict[str, list[str | re.Pattern]]):
        """
        Initializes the CsvColumns instance with the provided column definitions.
        Helps to parse csv files that have a tentative structure to them by finding column name matches.

        Parameters
        ----------
        columns : dict[str, list[str | re.Pattern]]
            The anticipated column names and their corresponding aliases or regex patterns.

        Example
        -------

        .. code-block:: python

            cols = cc.CsvColumns({
                'latitude': ['lat'],
                'datetime': ['UTC', 'localtime', re.compile(r"^dt")]
            })
            rows = ft.from_csv('Flight log', log_path, log_file_ext)
            cols.parse_header(rows[0])
            lat = float(rows[1][cols['latitude']])
            dt = datetime.fromisoformat(rows[1][cols['datetime']])
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.columns = {k: _ColumnHeader(k, columns[k], -1) for k in columns}

    @classmethod
    def SimpleColumns(cls, header_row: list[str]):
        """
        Creates a CsvColumns instance from a simple header row.

        This method initializes the columns using the header row as both the names and aliases.

        Parameters
        ----------
        header_row : list[str]
            A list of column names from the CSV header.

        Returns
        -------
        CsvColumns
            An instance of CsvColumns initialized with the provided header row.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        columns = {v: [v] for v in header_row}
        ret = cls(columns)
        ret.parse_header(header_row)
        return ret

    def parse_data_row(self, data_row: list[str], row_idx=-1):
        """
        Parses a data row and extracts values based on the matched column indices.

        Parameters
        ----------
        data_row : list[str]
            A list of values from a single row of the CSV file.
        row_idx : int, optional
            The index of the row being parsed, used for logging. Defaults to -1.

        Returns
        -------
        dict[str, str]
            A dictionary mapping column names to their corresponding values from the data row.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        ret: dict[str, str] = {}
        last_matched_idx = -1

        for column in self.columns.values():
            if column.idx >= 0 and len(data_row) > column.idx:
                ret[column.name] = data_row[column.idx]
                last_matched_idx = column.idx

        if last_matched_idx < len(data_row) - 1:
            if row_idx > -1:
                lt.debug(f"Found {len(data_row)-last_matched_idx-1} extra values in row {row_idx}")
            last_column = sorted(self.columns.values(), key=lambda c: c.idx)[-1]
            cnt = 2
            for i in range(last_matched_idx + 1, len(data_row)):
                ret[column.name + str(cnt)] = data_row[i]
                cnt += 1

        return ret

    def parse_header(
        self,
        header_row: list[str],
        error_on_not_found: bool | list[str] = True,
        ok_if_not_found: list[str] = None,
        alternatives: dict[str, list[str]] = None,
    ):
        """
        Parses the header row to find matches for the defined columns.

        This method updates the column indices based on the header row and checks for
        any missing columns, logging warnings or raising errors as specified.

        Parameters
        ----------
        header_row : list[str]
            A list of column names from the CSV header.
        error_on_not_found : bool | list[str], optional
            If True, raises an error for any missing columns. If a list, raises an error for columns in that list. Defaults to True.
        ok_if_not_found : list[str], optional
            A list of column names that are acceptable to be missing. Defaults to None.
        alternatives : dict[str, list[str]], optional
            A dictionary mapping column names to lists of alternative names. Defaults to None.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # add reverse values for the alternatives, if any
        if alternatives != None:
            ks = list(alternatives.keys())
            for k in ks:
                vs = alternatives[k]
                for v in vs:
                    if v not in alternatives:
                        alternatives[v] = []
                    alternatives[v].append(k)

        # find the column to header matches
        for hi, header in enumerate(header_row):
            slheader = header.strip().lower()

            # find the matching column for this header
            for k in self.columns:
                column = self.columns[k]
                if column.idx != -1:
                    continue

                # does this column match?
                found = False
                for alias in column.aliases:
                    if isinstance(alias, re.Pattern):
                        found = alias.match(slheader) != None
                    else:
                        found = alias.lower() in slheader
                    if found:
                        self.columns[k] = _ColumnHeader(column.name, column.aliases, hi)

        # debug: print the matched columns
        dbg_msg = "Parsed columns:\n"
        for k in self.columns:
            column = self.columns[k]
            if column.idx < 0:
                dbg_msg += f"    {column.name}: -1\n"
            else:
                dbg_msg += f"    {column.name}: {header_row[column.idx]} ({column.idx})\n"
        lt.debug(dbg_msg)

        # check that we found all the columns
        for k in self.columns:
            column = self.columns[k]
            if column.idx == -1:
                # we couldn't find this column:
                # 1) is it ok that it couldn't be found?
                # 2) does an alternative exist?
                # 3) log a warning or raise an error

                # 1
                if column.name in ok_if_not_found:
                    continue
                # 2
                if column.name in alternatives:
                    found = False
                    others = alternatives[column.name]
                    for alternative in others:
                        if alternative in self.columns and self.columns[alternative] != -1:
                            found = True
                            break
                    if found:
                        continue
                # 3
                msg = f"In CsvColumns.parse_header: couldn't find a match for the column '{column.name}' (aliases {column.aliases}) in the headers {header_row}!"
                if error_on_not_found == True or column.name in error_on_not_found:
                    lt.error_and_raise(RuntimeError, "Error: " + msg)
                else:
                    lt.warn(msg)

    def __getitem__(self, column_name) -> int:
        return self.columns[column_name].idx
