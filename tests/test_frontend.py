"""Frontend checks: the JS tests (node --test) and a guard against HTML-string sinks."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "app" / "static"

def test_markdown_renderer_js_tests():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js not installed")
    result = subprocess.run([node, "--test", "tests/js/*.test.js"], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

# Any of these turns a string into HTML; the UI must build DOM nodes instead (XSS)
HTML_SINKS = re.compile(r"\.(innerHTML|outerHTML)\s*[+]?=|insertAdjacentHTML|document\.write")

@pytest.mark.parametrize("path", sorted(STATIC.glob("*.html")) + sorted(STATIC.glob("*.js")), ids=lambda p: p.name)
def test_no_html_string_sinks(path):
    code = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                     if not line.lstrip().startswith("//"))  # comments may mention them
    assert not HTML_SINKS.search(code), f"{path.name} writes HTML from a string"
