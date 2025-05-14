import os
import time
from urllib.request import build_opener, install_opener, urlopen

# Needed for Cloudflare's firewall
opener = build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
install_opener(opener)


def download(src: str, dst: str, max_size: str = None) -> None:
    """
    Downloads a file from the provided source URL to the specified destination path
    only if the file doesn't already exist at the destination.

    Args:
        src (str): The URL of the file to download.
        dst (str): The local path where the file should be saved.
    """
    if os.path.exists(dst):
        return

    print("downloading %s -> %s..." % (src, dst))
    if max_size is not None:
        print("   stopping at %.2f MiB " % (int(max_size) / 2**20))

    t0 = time.time()
    outf = open(dst, "wb")
    inf = urlopen(src)
    info = dict(inf.info())
    content_size = int(info["Content-Length"])
    bs = 1 << 20
    totsz = 0

    while True:
        block = inf.read(bs)
        elapsed = time.time() - t0
        print(
            "  [%.2f s] downloaded %.2f MiB / %.2f MiB at %.2f MiB/s   "
            % (elapsed, totsz / 2**20, content_size / 2**20, totsz / 2**20 / elapsed),
            flush=True,
            end="\r",
        )
        if not block:
            break
        if max_size is not None and totsz + len(block) >= max_size:
            block = block[: max_size - totsz]
            outf.write(block)
            totsz += len(block)
            break
        outf.write(block)
        totsz += len(block)

    print("Download finished in %.2f s, total size %d bytes" % (time.time() - t0, totsz))


def replace(obj, **changes):
    """
    Create a new object of the same type as obj, replacing fields with values from changes.

    This function replicates the behavior of dataclasses.replace() for regular classes.

    Args:
        obj: The object to replace fields in
        **changes: Keyword arguments mapping field names to new values

    Returns:
        A new instance of the same type as obj with specified fields replaced

    Raises:
        TypeError: If an invalid field name is specified in changes
    """
    cls = obj.__class__

    for key in changes:
        if not hasattr(obj, key):
            raise TypeError(f"__init__() got an unexpected keyword argument '{key}'")

    field_values = {}
    for key, value in obj.__dict__.items():
        field_values[key] = value

    field_values.update(changes)
    return cls(**field_values)