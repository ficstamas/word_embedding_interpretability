import json
from multiprocessing.shared_memory import SharedMemory
import numpy as np


class Data:
    def __init__(self, memory_prefix: str):
        self.distance_matrix = memory_prefix+"distance_matrix"
        self.distance_matrix_shape = None
        self.transformed_space = memory_prefix+"transformed_space"
        self.transformed_space_shape = None
        self.transformed_space_distance_matrix = memory_prefix+"transformed_space_distance_matrix"
        self.transformed_space_distance_matrix_shape = None
        self._memories = []

    def free(self):
        for mem in self._memories:
            mem: SharedMemory
            mem.unlink()

    def init_shared_memory(self, name, shape, dtype, config):
        log = config.logger
        matrix = np.zeros(shape, dtype=dtype)
        memory = SharedMemory(name, create=True, size=matrix.nbytes)
        log.info(f"Memory allocated for {name} with shape {shape} and dtype {dtype}. Overall size in memory: "
                 f"{matrix.nbytes/1024/1024:.2f} MBytes ({matrix.nbytes} bytes)")
        del matrix
        self._memories.append(memory)

    def to_json(self) -> str:
        return json.dumps({
            "distance_matrix": self.distance_matrix,
            "transformed_space": self.transformed_space,
            "transformed_space_distance_matrix": self.transformed_space_distance_matrix,
        })

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return json.loads(self.to_json())
