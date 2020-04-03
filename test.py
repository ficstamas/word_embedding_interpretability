from interpretability.loader.embedding import embedding_reader
import numpy as np
from multiprocessing.shared_memory import SharedMemory
import json
import array

obj = embedding_reader("data/glove.6B.300d.txt", True, 5000)
shr = SharedMemory("embedding", create=False)
a = np.ndarray(obj.shape, buffer=shr.buf)
shr2 = SharedMemory("i2w", create=False)
b = array.array('B', shr2.buf)[:obj.i2w_memory_size]
c = b.tobytes().decode("utf8")
d = json.loads(c)
print("")
