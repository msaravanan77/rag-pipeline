"""Markdown parser built on python-frontmatter + marko.

Why marko (DECISIONS.md 1.1): it produces a typed AST where a heading is
Heading(level=2, children=[RawText("Configuring TLS")]) — level and text as
first-class fields. Heading-aware chunking is impossible with HTML-emitting
parsers like mistune without fragile HTML re-parsing.
"""

from __future__ import annotations

import re

import frontmatter as fm
import marko
from marko.block import FencedCode, Heading as MarkoHeading

from cncf_rag.ingestion.models import CodeBlock, Heading, ParsedDocument

# Hugo shortcodes appear throughout kubernetes/website and helm/helm-www,
# e.g. {{< note >}}, {{% capture body %}}, {{< codenew file="pods/pod.yaml" >}}.
# They are template directives, not content — embedding them adds pure noise.
_HUGO_SHORTCODE_RE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)

# HTML comments hold author notes and review markers in CNCF docs
# (e.g. <!-- overview -->, <!-- body --> structure markers in k8s docs).
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _ast_text(node: object) -> str:
    """Recursively collect raw text from a marko AST node's children."""
    if isinstance(node, str):
        return node
    children = getattr(node, "children", None)
    if children is None:
        return ""
    if isinstance(children, str):
        return children
    return "".join(_ast_text(child) for child in children)


class MarkdownParser:
    """Parses a raw CNCF Markdown file into structured content.

    Output contract: `content` is the cleaned body (no frontmatter, no Hugo
    shortcodes, no HTML comments); `headings` and `code_blocks` carry
    char_offset positions valid within that cleaned content.
    """

    def __init__(self) -> None:
        self._md = marko.Markdown()

    def parse(self, raw_content: str, source_path: str) -> ParsedDocument:
        """Split frontmatter, clean the body, and extract structure from the AST."""
        post = fm.loads(raw_content)
        body = post.content

        # Cleaning happens BEFORE AST extraction so char offsets recorded from
        # the AST line positions match the content string chunkers will slice.
        body = _HUGO_SHORTCODE_RE.sub("", body)  # Hugo template directives ({{< note >}} etc.)
        body = _HTML_COMMENT_RE.sub("", body)  # k8s structural markers (<!-- overview -->)

        headings, code_blocks = self._extract_structure(body)
        return ParsedDocument(
            content=body,
            headings=headings,
            code_blocks=code_blocks,
            frontmatter=dict(post.metadata),
            source_path=source_path,
        )

    def _extract_structure(self, body: str) -> tuple[list[Heading], list[CodeBlock]]:
        """Walk the marko AST collecting typed heading and code block nodes.

        char_offset is computed from the node's line number against the body's
        line-start offsets — marko tracks source lines but not char positions.
        """
        doc = self._md.parse(body)
        line_offsets = [0]
        for line in body.splitlines(keepends=True):
            line_offsets.append(line_offsets[-1] + len(line))

        def offset_for(node: object) -> int:
            # marko line numbers are 1-based; fall back to 0 for synthetic nodes.
            line = getattr(node, "line_number", None) or 1
            idx = min(line - 1, len(line_offsets) - 1)
            return line_offsets[idx]

        headings: list[Heading] = []
        code_blocks: list[CodeBlock] = []
        for node in doc.children:
            if isinstance(node, MarkoHeading):
                headings.append(
                    Heading(level=node.level, text=_ast_text(node).strip(), char_offset=offset_for(node))
                )
            elif isinstance(node, FencedCode):
                code_blocks.append(
                    CodeBlock(
                        language=node.lang or "",
                        content=_ast_text(node),
                        char_offset=offset_for(node),
                    )
                )
        # marko's line_number attribute is only present on some versions; if all
        # offsets collapsed to 0, recover heading offsets by literal text search.
        if headings and all(h.char_offset == 0 for h in headings):
            cursor = 0
            for h in headings:
                marker = "#" * h.level + " " + h.text
                pos = body.find(marker, cursor)
                if pos >= 0:
                    h.char_offset = pos
                    cursor = pos + len(marker)
        return headings, code_blocks
