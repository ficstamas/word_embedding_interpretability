from .general import *
from .whitening import Whiten

TRANSFORM_MAP = {
    "std": Standardize,
    "norm": Normalize,
    "center": Centering,
    "whiten": Whiten
}
