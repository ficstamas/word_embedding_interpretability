from interpretability.utils.metaclasses import Singleton
from typing import Literal
import os
import json
from .config_modules import *
import logging
import sys
import platform
import time


__all__ = ["Config"]


class Config(metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        if 'access' not in kwargs or 'access' in kwargs and not kwargs['access']:
            if kwargs['memory_prefix'] is None:
                # Setting up shared memory object prefixes in a platform dependent way
                if platform.system() == 'Linux':
                    self.memory_prefix = f"{os.getuid()}_{os.getpid()}_{int(round(time.time() * 1000))}_"
                elif platform.system() == 'Windows':
                    self.memory_prefix = f"{int(round(time.time() * 1000))}_"
                else:
                    print("The OS is not supported")
                    sys.exit(0)
            else:
                self.memory_prefix = kwargs['memory_prefix']
            self.embedding = Embedding()
            self.semantic_categories = SemanticCategories()
            self.distance = Distance()
            self.kde = KDE()
            self.project = Project()
            self.data = Data(self.memory_prefix)
            self.model = ModelParams()
            # Logging config
            self.logger = logging.getLogger("default")

    def set_embedding(self, path: str, dense: bool, lines_to_read: int):
        self.embedding.path = path
        self.embedding.dense = dense
        self.embedding.lines_to_read = lines_to_read

    def set_semantic_categories(self, path: str, load_method: Literal['semcat', 'old_semcat', 'semcor'],
                                drop_method: Literal['random', 'category_center'],
                                drop_rate: float, seed: int):
        self.semantic_categories.path = path
        self.semantic_categories.load_method = load_method
        self.semantic_categories.drop_method = drop_method
        self.semantic_categories.drop_rate = drop_rate
        self.semantic_categories.seed = seed
        if self.semantic_categories.path[-5:] == ".json":
            self.semantic_categories.file_format = "json"
        else:
            self.semantic_categories.file_format = "dict"

    def set_project_path(self, path: str, name: str):
        self.project.workspace = os.path.join(os.getcwd(), path)
        self.project.name = name

        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

        path = os.path.join(self.project.logs, "debug.log")
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        error_file_handler = logging.FileHandler(os.path.join(self.project.logs, "error.log"))
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        # self.logger.addHandler(stream_handler)
        self.logger.addHandler(error_file_handler)
        self.logger.info("Logging Handlers are initiated!")

    def set_kde(self, kernel: str, bandwidth: float):
        self.kde.kernel = kernel
        self.kde.bandwidth = bandwidth

    def set_distance(self, dist: Literal['bhattacharyya', 'hellinger', 'bhattacharyya_normal', 'hellinger_normal',
                                         'hellinger_exponential', 'bhattacharyya_exponential']):
        self.distance.name = dist

    def log_config(self):
        log = self.logger
        log.info(f"Workspace is set to: {self.project.project}")
        log.info(f"The project name is: {self.project.name}")
        log.info(f"Embedding path is: {self.embedding.path}" + (" which is a dense" if self.embedding.dense else " which is a sparse") + " embedding.")
        log.info(f"{self.embedding.lines_to_read} are read")
        log.info(f"{self.semantic_categories.path} is going to be used with {self.semantic_categories.drop_method} "
                 f"method, {self.semantic_categories.drop_rate} drop rate and {self.semantic_categories.seed} seed")
        log.info(f"{self.distance.name} is going to be used")
        log.info(f"Kernel Density Estimation: kernel={self.kde.kernel}, bandwidth={self.kde.bandwidth}")
        log.info(f"The program is going to use {self.project.processes} processes")
        log.info(f"MCMC model params {self.model.mcmc_acceptance} acceptance and {self.model.mcmc_noise} noise")

    def to_json(self):
        return json.dumps(
            {
                "embedding": self.embedding.__repr__(),
                "project": self.project.__repr__(),
                "kde": self.kde.__repr__(),
                "distance": self.distance.__repr__(),
                "semantic_categories": self.semantic_categories.__repr__(),
                "model_params": self.model.__repr__(),
                "data": self.data.__repr__()
            })

    def restore_from_json(self, path):
        self.logger.info(f"Restoring config from JSON...")
        if not os.path.exists(path):
            self.logger.error(f"{path} does not exist")
            sys.exit(-1)
        fp = open(path)
        params = json.load(fp)
        fp.close()
        self.project.from_dict(params["project"])
        self.embedding.from_dict(params["embedding"])
        self.kde.from_dict(params["kde"])
        self.distance.from_dict(params["distance"])
        self.semantic_categories.from_dict(params["semantic_categories"])
        self.model.from_dict(params["model_params"])
        try:
            self.data.from_dict(params["data"])
        except (NameError, KeyError):
            if not self.embedding.path.find("_semcor.") == -1:
                self.data.test_word_weights_path = self.embedding.path.replace("_semcor.", "_ALL.")
            else:
                self.data.test_word_weights_path = self.embedding.path.replace("semcor.", "ALL.")
        self.logger.info("Config restored!")

    def free(self):
        self.embedding.embedding.free()
        self.data.free()

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return json.loads(self.__str__())
