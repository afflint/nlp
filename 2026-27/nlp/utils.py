import textwrap
from pprint import pprint


def show(obj, width: int = 100):
    """Print an object so that it fits the notebook width, without horizontal scrolling.

    Strings are printed as plain text, wrapped on whitespace (existing line breaks are kept).
    Any other object (lists of tokens, dicts, ...) is printed with pprint, several items per line.
    """
    if isinstance(obj, str):
        for line in obj.splitlines() or [""]:
            print(textwrap.fill(line, width=width) if line.strip() else "")
    else:
        pprint(obj, width=width, compact=True, sort_dicts=False)
