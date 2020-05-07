from interpretability.loader.embedding import embedding_reader
import numpy as np
from multiprocessing.shared_memory import SharedMemory
import json
import array

from interpretability.core.config import Config
import pprint
# obj = embedding_reader("data/glove.6B.300d.txt", True, 5000)
#
# # embedding
# shr = SharedMemory(obj.embedding_memory_name, create=False)
# a = np.ndarray(obj.embedding_memory_shape, buffer=shr.buf)
#
# # i2w
# shr2 = SharedMemory(obj.i2w_memory_name, create=False)
# i2w = obj.buff_to_dict(shr2, obj.i2w_memory_size)

conf = Config()

conf.embedding.path = "asd/emb.txt"
conf.set_distance("bhattacharyya")

print(conf.distance)

conf.set_project_path("./", "test")

print(conf.project)

conf.log_config()
