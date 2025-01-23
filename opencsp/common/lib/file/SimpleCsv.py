import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.file.CsvColumns as csv
import opencsp.common.lib.file.CsvInterface as csvi


class SimpleCsv:
    """
    A class for simple parsing of CSV files.

    This class allows for reading a CSV file and provides methods to access
    the header, columns, and rows of the file in a structured manner.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __init__(self, description: str, file_path: str, file_name_ext: str):
        """
        Initializes the SimpleCsv instance and parses the CSV file.

        Parameters
        ----------
        description : str
            A description of the file to be processed, or None to suppress output to stdout.
        file_path : str
            The path to the CSV file to be processed.
        file_name_ext : str
            The name and extension of the CSV file to be processed.

        Raises
        ------
        FileNotFoundError
            If the specified CSV file does not exist.
        ValueError
            If the CSV file is empty or improperly formatted.

        Example
        -------

        .. code-block:: python

            parser = scsv.SimpleCsv("example file", file_path, file_name_ext)
            for row_dict in parser:
                print(row_dict)
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.description = description
        self.file_path = file_path
        self.file_name_ext = file_name_ext

        lines = ft.from_csv(description, file_path, file_name_ext)
        header_row = lines[0]
        self.cols = csv.CsvColumns.SimpleColumns(header_row)

        self.rows: list[dict[str, str]] = []
        for row in lines[1:]:
            self.rows.append(self.cols.parse_data_row(row))

    def get_header(self):
        """
        Returns the header of the CSV file as a comma-separated string.

        Returns
        -------
        str
            A string representation of the header row of the CSV file.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return ",".join(self.get_columns())

    def get_columns(self):
        """
        Returns a list of column names from the CSV file.

        Returns
        -------
        list[str]
            A list of column names extracted from the CSV header.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return [col.name for col in self.cols.columns.values]

    def get_rows(self):
        """
        Returns the rows of the CSV file as a list of dictionaries.

        Each dictionary corresponds to a row in the CSV file, with column names as keys.

        Returns
        -------
        list[dict[str, str]]
            A list of dictionaries representing the rows of the CSV file.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return self.rows

    def __iter__(self):
        return self.rows.__iter__()
