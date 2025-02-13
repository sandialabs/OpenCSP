import dataclasses
import re

import opencsp.common.lib.tool.log_tools as lt


@dataclasses.dataclass
class Match:
    lineno: int
    colno: int
    colend: int
    line: str
    line_part: str
    matcher: "SensitiveStringMatcher"
    msg: str = ""


class SensitiveStringMatcher:
    def __init__(self, name: str, *patterns: str):
        self.name = name
        self.patterns: list[re.Pattern | str] = []
        self.neg_patterns: list[re.Pattern | str] = []
        self.log_type = lt.log.DEBUG
        self.log = lt.debug
        self.case_sensitive = False

        next_is_regex = False
        all_regex = False
        remaining_negative_match = False
        for pattern in patterns:
            if pattern.startswith("**"):
                directive = pattern[2:]
                if directive == "debug":
                    self.log_type = lt.log.DEBUG
                    self.log = lt.debug
                elif directive == "info":
                    self.log_type = lt.log.INFO
                    self.log = lt.info
                elif directive == "warn":
                    self.log_type = lt.log.WARN
                    self.log = lt.warn
                elif directive == "error":
                    self.log_type = lt.log.ERROR
                    self.log = lt.error
                elif directive == "next_is_regex":
                    next_is_regex = True
                elif directive == "all_regex":
                    all_regex = True
                elif directive == "case_sensitive":
                    self.case_sensitive = True
                elif directive == "dont_match":
                    remaining_negative_match = True

            else:
                if next_is_regex or all_regex:
                    pattern = re.compile(pattern)
                else:
                    pass

                if not remaining_negative_match:
                    self.patterns.append(pattern)
                else:
                    self.neg_patterns.append(pattern)

                next_is_regex = False

        # case insensitive matching
        if not self.case_sensitive:
            for patterns in [self.patterns, self.neg_patterns]:
                for i, pattern in enumerate(patterns):
                    if isinstance(pattern, str):
                        patterns[i] = pattern.lower()
                    else:
                        p: re.Pattern = pattern
                        patterns[i] = re.compile(p.pattern.lower())

    def _search_pattern(self, ihaystack: str, pattern: re.Pattern | str) -> None | list[int]:
        if isinstance(pattern, str):
            # Check for occurances of string literals
            if pattern in ihaystack:
                start = ihaystack.index(pattern)
                end = start + len(pattern)
                return [start, end]

        else:
            # Check for instances of regex matches
            re_match = pattern.search(ihaystack)
            if re_match:
                start, end = re_match.span()[0], re_match.span()[1]
                return [start, end]

        return None

    def _search_patterns(self, ihaystack: str, patterns: list[re.Pattern | str]) -> dict[re.Pattern | str, list[int]]:
        ret: dict[re.Pattern | str, list[int]] = {}

        for pattern in patterns:
            span = self._search_pattern(ihaystack, pattern)
            if span:
                ret[pattern] = span

        return ret

    def check_lines(self, lines: list[str]):
        matches: list[Match] = []

        for lineno, line in enumerate(lines):
            iline = line if self.case_sensitive else line.lower()

            # Check for matching patterns in this line
            possible_matching = self._search_patterns(iline, self.patterns)

            # Filter out negative matches in the matching patterns
            matching: dict[re.Pattern | str, list[int]] = {}
            for pattern in possible_matching:
                span = possible_matching[pattern]
                line_part = iline[span[0] : span[1]]
                if len(self._search_patterns(line_part, self.neg_patterns)) == 0:
                    matching[pattern] = span

            # Register the matches
            for pattern in matching:
                span = matching[pattern]

                start, end = span[0], span[1]
                line_part = line[start:end]
                line_context = f"`{line_part}`"
                if start > 0:
                    line_context = line[max(start - 5, 0) : start] + line_context
                if end < len(line):
                    line_context = line_context + line[end : min(end + 5, len(line))]

                match = Match(lineno + 1, start, end, line, line_part, self)
                self.set_match_msg(match, pattern, line_context)
                matches.append(match)
                self.log(match.msg)

        return matches

    def set_match_msg(self, match: Match, pattern: re.Pattern | str, line_context: str):
        log_msg = (
            f"'{self.name}' string matched to pattern '{pattern}' on line {match.lineno} "
            + f'[{match.colno}:{match.colend}]: "{line_context.strip()}" ("{match.line.strip()}")'
        )
        match.msg = log_msg
