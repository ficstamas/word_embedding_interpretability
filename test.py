from interpretability.loader.embedding import embedding_reader
import numpy as np
from multiprocessing.shared_memory import SharedMemory
import json
import array

obj = embedding_reader("data/glove.6B.300d.txt", True, 5000)

# embedding
shr = SharedMemory(obj.embedding_memory_name, create=False)
a = np.ndarray(obj.embedding_memory_shape, buffer=shr.buf)

# i2w
shr2 = SharedMemory(obj.i2w_memory_name, create=False)
i2w = obj.buff_to_dict(shr2, obj.i2w_memory_size)

print("")
