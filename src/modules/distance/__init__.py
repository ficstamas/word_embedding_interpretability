from .bhattacharyya import *
from .hellinger import *

DISTANCE_MAP = {
    "hellinger": continuous_hellinger_distance,
    "bhattacharyya": continuous_bhattacharyya_distance,
    "hellinger_normal": closed_hellinger_distance,
    "bhattacharyya_normal": closed_bhattacharyya_distance,
    "hellinger_exponential": exponential_hellinger_distance,
    "bhattacharyya_exponential": exponential_bhattacharyya_distance
}
