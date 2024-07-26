from typing import Optional


class CsvInterface:
    """Template class for storing data in comma-separated-value files. Inheriting classes include methods to convert to/from csv lines.

    Required Methods
    ----------------
        csv_header: Callable[[str],str]
                    Static method. Takes at least one parameter 'delimeter' and returns the string that represents the csv header.
        to_csv_line: Callable[[str], str]
                     Takes at least one parameter 'delimeter' and returns the string that represents the instance of that class.
        from_csv_line: Callable[[list[str]], tuple[Any,list[str]]]
                       Class method. Takes at least the parameter 'data' and returns an instance of the class and the remaining data strings.

    Optional Methods
    ----------------
        to_csv: Callable[[str,str,str,bool], None]
                Optional. Takes the description, file_path, file_name_ext, and error_if_dir_not_exist parameters and writes the header and any rows to a file.
        from_csv: Callable[[str,str,str], list[Any]]
                  Optional. Class method. Takes the description, file_path, and file_name_ext parameters and returns a list of class instances from that file.
    """

    # TODO add decorator to wrap dataclasses.dataclass classes with an automatic CsvInterface

    @staticmethod
    def csv_header(delimeter=",") -> str:
        """Return a simple string which can be used as the header line in a csv file."""
        raise NotImplementedError()

    def to_csv_line(self, delimeter=",") -> str:
        """Return a string representation of this instance, to be written to a csv file. Does not include a trailing newline."""
        raise NotImplementedError()

    @classmethod
    def from_csv_line(cls, data: list[str]) -> tuple["CsvInterface", list[str]]:
        """Construct an instance of this class from the pre-split csv line 'data'. Also return any leftover portion of the csv line that wasn't used."""
        raise NotImplementedError()

    def to_csv(
        self,
        description: str,
        file_path: str,
        file_name: str,
        error_if_dir_not_exist: bool = True,
        rows: Optional[list["CsvInterface"]] = None,
        overwrite=False,
    ):
        """Create a csv file with a header and one or more lines (one line per contained instance if this is a collection of CsvInterface objects).

        If rows is not None, then write all of the given rows with the to_csv_line() method.
        If rows is None, then attempt to use self.rows or self.rows() if it exists to write all the contained rows.

        This is the basic implementation of to_csv. Subclasses are encouraged to extend this method.
        """
        # get the data lines
        row_strs: list[str | "CsvInterface"] = []
        if rows == None:
            if hasattr(self, 'rows'):
                if callable(self.rows):
                    rows = self.rows()
                else:
                    rows = self.rows
                row_strs = rows
            else:
                row_strs = [self.to_csv_line()]
        else:
            for row in rows:
                if isinstance(row, CsvInterface):
                    row_strs.append(row.to_csv_line())
                elif isinstance(row, str):
                    row_strs.append(row)

        # get the header line
        header_str = self.csv_header()
        if len(row_strs) > 0 and isinstance(row_strs[0], CsvInterface):
            header_str = row_strs[0].csv_header()

        import opencsp.common.lib.tool.file_tools as ft

        ft.to_csv(
            description,
            file_path,
            file_name,
            error_if_dir_not_exist=error_if_dir_not_exist,
            heading_line=header_str,
            data_lines=row_strs,
            overwrite=overwrite,
        )

    @classmethod
    def from_csv(cls, description: str, file_path: str, file_name_ext: str):
        """Return N instances of this class from a csv file with a header and N lines.

        Basic implementation of from_csv. Subclasses are encouraged to extend this method.
        """

        import opencsp.common.lib.tool.file_tools as ft

        data_rows = ft.from_csv(description, file_path, file_name_ext)

        return [cls.from_csv_line(row) for row in data_rows[1:]]
