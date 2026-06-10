"""Content normalization applied after parsing, before chunking.

Order matters and is fixed: unicode normalization first (so later regexes see
canonical forms), structural noise removal last. Code blocks are protected
throughout — a normalization that altered YAML indentation would corrupt
manifests that the system promises to reproduce verbatim (system prompt rule 6).
"""

from __future__ import annotations

import re
import unicodedata

_HUGO_SHORTCODE_RE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_FENCE_RE = re.compile(r"(```.*?```|~~~.*?~~~)", re.DOTALL)


class Preprocessor:
    """Applies the fixed cleaning sequence, preserving fenced code verbatim."""

    def clean(self, content: str) -> str:
        # Split out fenced code blocks first; every transformation below runs
        # only on the prose segments so code is preserved byte-for-byte.
        segments = _FENCE_RE.split(content)
        cleaned: list[str] = []
        for i, segment in enumerate(segments):
            if i % 2 == 1:  # odd indices are the captured fenced blocks
                cleaned.append(segment)
                continue
            text = segment
            # NFC: composes é from e+combining-accent — k8s docs mix both forms,
            # and inconsistent forms make identical strings embed differently.
            text = unicodedata.normalize("NFC", text)
            # CRLF from Windows-authored contributions → LF.
            text = text.replace("\r\n", "\n")
            # Hugo shortcodes survive here when frontmatter splitting reordered
            # content (defense in depth — parser already strips most).
            text = _HUGO_SHORTCODE_RE.sub("", text)
            text = _HTML_COMMENT_RE.sub("", text)
            # Collapse 3+ blank lines: shortcode removal leaves gaps that would
            # otherwise become token-wasting whitespace in chunks.
            text = _BLANK_LINES_RE.sub("\n\n", text)
            # Trailing whitespace per line.
            text = "\n".join(line.rstrip() for line in text.split("\n"))
            cleaned.append(text)
        return "".join(cleaned)
