"""Build the GitHub Pages site: in-browser (WebAssembly) marimo notebooks.

Usage: uv run python scripts/build_site.py [output_dir]

Exports the solution notebooks (source/) and the exercise notebooks (release/)
into one directory so they share a single copy of marimo's assets.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LECTURES = [
    ("lecture1_fd", "lecture1", "Lecture 1 — Finite differences and the method of lines"),
    ("lecture2_dg", "lecture2", "Lecture 2 — Nodal discontinuous Galerkin methods"),
]
VERSIONS = [("exercises", "release"), ("solutions", "source")]

# marimo's WASM export embeds its default user config, which does not run the
# notebook on startup in edit mode. Students would then have to find the
# "run all" button before seeing anything, so switch autorun on.
AUTORUN_OFF = '"auto_instantiate": false'
AUTORUN_ON = '"auto_instantiate": true'


def export(notebook: Path, output: Path) -> None:
    subprocess.run(
        ["marimo", "export", "html-wasm", str(notebook), "-o", str(output), "--mode", "edit", "-f"],
        check=True,
    )
    html = output.read_text()
    if html.count(AUTORUN_OFF) != 1:
        sys.exit(f"{output}: expected exactly one {AUTORUN_OFF!r}; marimo's export format changed")
    output.write_text(html.replace(AUTORUN_OFF, AUTORUN_ON))


def index_page() -> str:
    rows = "\n".join(
        f"""      <tr>
        <td>{title}</td>
        <td><a href="{slug}-exercises.html">exercises</a></td>
        <td><a href="{slug}-solutions.html">solutions</a></td>
      </tr>"""
        for _, slug, title in LECTURES
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Introduction to solving PDEs</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 48rem; margin: 3rem auto; padding: 0 1rem; line-height: 1.5; }}
    table {{ border-collapse: collapse; }}
    td {{ padding: 0.4rem 1rem 0.4rem 0; }}
  </style>
</head>
<body>
  <h1>Introduction to solving PDEs</h1>
  <p>Interactive marimo notebooks that run entirely in your browser; nothing to install.</p>
  <table>
{rows}
  </table>
  <p>The first load downloads Python and the scientific packages, which can take up to a minute.
  Nothing you type is stored on a server: to keep your work, download the notebook from the menu at the top right (Export&hellip;).</p>
  <p>Source: <a href="https://github.com/nilsdeppe/pde-intro">github.com/nilsdeppe/pde-intro</a></p>
</body>
</html>
"""


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "_site").resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, slug, _ in LECTURES:
        for version, directory in VERSIONS:
            export(ROOT / directory / name / f"{name}.py", out / f"{slug}-{version}.html")
    (out / "index.html").write_text(index_page())
    (out / ".nojekyll").touch()
    print(f"Site written to {out}")


if __name__ == "__main__":
    main()
