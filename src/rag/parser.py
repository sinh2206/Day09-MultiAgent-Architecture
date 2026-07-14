from __future__ import annotations


def parse_policy_markdown(markdown_text: str) -> list[dict[str, str]]:
    """Split policy into the rubric-required H2 + H3 + content chunks."""
    chunks: list[dict[str, str]] = []
    current_h2 = ""
    current_h3 = ""
    content_lines: list[str] = []

    def flush() -> None:
        nonlocal content_lines
        content = "\n".join(content_lines).strip()
        if current_h2 and current_h3 and content:
            chunks.append(
                {
                    "section_h2": current_h2,
                    "section_h3": current_h3,
                    "content": content,
                    "citation": (
                        f"policy_mock_vi.md > {current_h2} > {current_h3}"
                    ),
                    "rendered_text": (
                        f"## {current_h2}\n### {current_h3}\n{content}"
                    ),
                }
            )
        content_lines = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## ") and not line.startswith("### "):
            flush()
            current_h2 = line[3:].strip()
            current_h3 = ""
        elif line.startswith("### "):
            flush()
            current_h3 = line[4:].strip()
        elif current_h3:
            content_lines.append(line)
    flush()
    return chunks
