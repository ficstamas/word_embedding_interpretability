from interpretability.utils.metaclasses import Singleton
from interpretability.distance import bhattacharyya, hellinger
from typing import Literal
import os
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class Config(metaclass=Singleton):
    DISTANCES = {
        'bhattacharyya': bhattacharyya.continuous_bhattacharyya_distance,
        'hellinger': hellinger.continuous_hellinger_distance,
        'bhattacharyya_normal': bhattacharyya.closed_bhattacharyya_distance,
        'hellinger_normal': hellinger.closed_hellinger_distance
    }

    def __init__(self):
        super(Config, self).__init__()
        self.params = {
            # Embedding parameters
            "embedding": {
                "path": None,
                "dense": None,
                "lines_to_read": None
            },
            # Params of semantic categories
            "semantic_categories": {
                "path": None,
                "drop_method": None,
                "drop_rate": None,
                "seed": None
            },
            # Path to project directory
            "project": {
                "workspace": None,
                "name": None,
                "project": None,
                "logs": None,
                "models": None,
                "results": None
            },
            # Kernel Density Estimation parameters
            "kde": {
                "kernel": None,
                "bandwidth": None
            },
            "distance": {
                "name": None,
                "function": None
            }
        }

    def __getitem__(self, item):
        return self.params[item]

    def set_embedding(self, path: str, dense: bool, lines_to_read: int):
        emb: dict
        emb = self.params["embedding"]
        emb["path"] = path
        emb["dense"] = dense
        emb["lines_to_read"] = lines_to_read

    def set_semantic_categories(self, path: str, drop_method: Literal['random', 'category_center'],
                                drop_rate: float, seed: int):
        semcat: dict
        semcat = self.params["embedding"]
        semcat["path"] = path
        semcat["drop_method"] = drop_method
        semcat["drop_rate"] = drop_rate
        semcat["seed"] = seed

    def set_project_path(self, path: str, name: str):
        self.params["project"]["workspace"] = os.path.join(os.getcwd(), path)
        self.params["project"]["name"] = name
        base, logs, models, results = self.generate_directory_structure(self.params["project"]["workspace"], name)
        self.params["project"]["workspace"] = base
        self.params["project"]["logs"] = logs
        self.params["project"]["models"] = models
        self.params["project"]["results"] = results

    def set_kde(self, kernel: str, bandwidth: float):
        kde: dict
        kde = self.params["project"]
        kde["kernel"] = kernel
        kde["bandwidth"] = bandwidth

    def set_distance(self, distance: Literal['bhattacharyya', 'hellinger', 'bhattacharyya_normal', 'hellinger_normal']):
        self.params["distance"]["name"] = distance
        self.params["distance"]["function"] = self.DISTANCES[distance]

    def distance(self):
        return self.params["distance"]["function"]

    @classmethod
    def generate_directory_structure(cls, path: str, name: str) -> tuple:
        """
        Generates the file structure
        :param path: Path to workspace
        :param name: Name of the project
        :return: Returns the path to the project, logs, models, results
        """
        project_base = os.path.join(path, name)
        logging.info(f"Generating directory structure in: {path}")
        try:
            os.mkdir(project_base)
            logging.info(f"Project directory created at: {project_base}")
        except FileExistsError:
            logging.info(f"Project directory exists in: {project_base}")
            logging.info(f"Every file is going to be overwritten in this directory!")
        # generating dirs

        project_logs = os.path.join(project_base, "logs")
        project_models = os.path.join(project_base, "models")
        project_results = os.path.join(project_base, "results")
        try:
            os.mkdir(project_logs)
            logging.info(f"Logging directory created at: {project_logs}")
            os.mkdir(project_models)
            logging.info(f"Models directory created at: {project_models}")
            os.mkdir(project_results)
            logging.info(f"Results directory created at: {project_results}")
        except FileExistsError:
            pass

        return project_base, project_logs, project_models, project_results
