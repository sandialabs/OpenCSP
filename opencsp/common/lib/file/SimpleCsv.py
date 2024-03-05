import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.file.CsvColumns as csv
import opencsp.common.lib.file.CsvInterface as csvi


class SimpleCsv:
    def __init__(self, description: str, file_path: str, file_name_ext: str):
        """Allows for simple CSV file parsing.

        Example::

            parser = scsv.SimpleCsv("example file", file_path, file_name_ext)
            for row_dict in parser:
                print(row_dict)

        Parameters:
        -----------
            - description (str): A description of the file to be processed, or None to not print to stdout.
            - file_path (str): Path to the file to be processed.
            - file_name_ext (str): Name and extension of the file to be processed.
        """
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
        return ",".join(self.get_columns())

    def get_columns(self):
        return [col.name for col in self.cols.columns.values]

    def get_rows(self):
        return self.rows

    def __iter__(self):
        return self.rows.__iter__()
