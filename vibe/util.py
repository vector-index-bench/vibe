import os
import tempfile
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
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return

    print("downloading %s -> %s..." % (src, dst))
    size_limit = int(max_size) if max_size is not None else None
    if max_size is not None:
        print("   stopping at %.2f MiB " % (size_limit / 2**20))

    t0 = time.time()
    bs = 1 << 20
    totsz = 0
    temp_path = None

    try:
        with urlopen(src) as inf:
            content_length = inf.info().get("Content-Length")
            content_size = int(content_length) if content_length is not None else None

            dst_dir = os.path.dirname(os.path.abspath(dst))
            fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(dst)}.", suffix=".download", dir=dst_dir)

            with os.fdopen(fd, "wb") as outf:
                while True:
                    block = inf.read(bs)
                    elapsed = max(time.time() - t0, 1e-9)
                    if content_size is None:
                        progress = "  [%.2f s] downloaded %.2f MiB at %.2f MiB/s   " % (
                            elapsed,
                            totsz / 2**20,
                            totsz / 2**20 / elapsed,
                        )
                    else:
                        progress = "  [%.2f s] downloaded %.2f MiB / %.2f MiB at %.2f MiB/s   " % (
                            elapsed,
                            totsz / 2**20,
                            content_size / 2**20,
                            totsz / 2**20 / elapsed,
                        )
                    print(progress, flush=True, end="\r")

                    if not block:
                        break
                    if size_limit is not None and totsz + len(block) >= size_limit:
                        block = block[: size_limit - totsz]
                        outf.write(block)
                        totsz += len(block)
                        break
                    outf.write(block)
                    totsz += len(block)

        os.replace(temp_path, dst)
        temp_path = None
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)

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
