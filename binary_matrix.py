import numpy as np


_magic_dtype = {
        0x1E3D4C51: ('float32', 4),
        0x1E3D4C53: ('float64', 8),
        0x1E3D4C54: ('int32', 4),
        0x1E3D4C55: ('uint8', 1),
        0x1E3D4C56: ('int16', 2),
        }
_dtype_magic = {
        'float32': 0x1E3D4C51,
        'float64': 0x1E3D4C53,
        'int32': 0x1E3D4C54,
        'uint8': 0x1E3D4C55,
        'int16': 0x1E3D4C56
        }


def read_int32(file_):
    """
    """
    x = file_.read(4)
    x = np.fromstring(x, dtype="int32").item()
    return x


def read_header(file_):
    """
    """
    magic = read_int32(file_)
    magic_t, el_size = _magic_dtype[magic]

    ndim = read_int32(file_)
    dims = np.fromfile(file_, dtype="int32", count=max(ndim, 3))[:ndim]

    return magic_t, el_size, ndim, dims


def read(file_):
    """
    """
    magic_t, el_size, ndim, dims = read_header(file_)

    data = np.fromfile(file_, dtype=magic_t, count=dims.prod())
    data = data.reshape(dims)

    return data


def load(filename):
    """
    """
    with open(filename, "rb") as file_:
        data = read(file_)

    return data
