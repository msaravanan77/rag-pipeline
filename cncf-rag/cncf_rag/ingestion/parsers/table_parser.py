"""Markdown table extraction.

API reference pages (e.g. kubectl flag tables, Helm chart value tables) carry
most of their information in tables. Extracting them as structured rows lets
chunkers keep a table row together with its header — a row without its header
("`--dry-run` | bool | false") is meaningless after retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_SEPARATOR_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


@dataclass
class MarkdownTable:
    headers: list[str]
    rows: list[list[str]]
    char_offset: int


class TableParser:
    """Finds GitHub-flavored Markdown tables and returns them as header+rows."""

    def extract_tables(self, content: str) -> list[MarkdownTable]:
        tables: list[MarkdownTable] = []
        lines = content.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))

        i = 0
        while i < len(lines) - 1:
            header_match = _TABLE_ROW_RE.match(lines[i])
            if header_match and _SEPARATOR_RE.match(lines[i + 1]):
                headers = [cell.strip() for cell in header_match.group(1).split("|")]
                rows: list[list[str]] = []
                j = i + 2
                while j < len(lines):
                    row_match = _TABLE_ROW_RE.match(lines[j])
                    if not row_match:
                        break
                    rows.append([cell.strip() for cell in row_match.group(1).split("|")])
                    j += 1
                tables.append(MarkdownTable(headers=headers, rows=rows, char_offset=offsets[i]))
                i = j
            else:
                i += 1
        return tables
