def non_blank_lines(f):
    """
    Generator which returns only non-empty (ignoring whitespace) lines from a filehandle
    :param f: file handle
    :return:
    """
    for l in f:
        line = l.rstrip()
        if line:
            yield line
from datetime import datetime


def current_timestamp() -> str:
    """
    Returns the current time in YYYY-MM-DD HH:MM:SS format\n
    @:return current time as string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
