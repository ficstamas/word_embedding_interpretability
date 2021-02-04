import logging, os
import json
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class Project:
    def __init__(self):
        self._workspace = None
        self._name = None
        self.project = None
        self.logs = None
        self.models = None
        self.results = None
        self.processes = None
        self._overwrite = False
        self._structure_generated = False

    @property
    def workspace(self):
        return self._workspace

    @property
    def name(self):
        return self._name

    @property
    def overwrite(self):
        return self._overwrite

    @overwrite.setter
    def overwrite(self, val=False):
        self._overwrite = val

    @workspace.setter
    def workspace(self, val):
        if self.name is not None and not self._structure_generated:
            self._workspace = val
            self.generate_directory_structure()
            self._structure_generated = True
        self._workspace = val

    @name.setter
    def name(self, val):
        if self.workspace is not None and not self._structure_generated:
            self._name = val
            self.generate_directory_structure()
            self._structure_generated = True
        self._name = val

    def generate_directory_structure(self):
        """
        Generates the file structure
        """
        try:
            os.makedirs(self.workspace, exist_ok=True)
            logging.info(f"Workspace created at: {self.workspace}")
        except FileExistsError:
            pass
        self.project = os.path.join(self.workspace, self.name)
        logging.info(f"Creating project in: {self.name}")
        try:
            os.mkdir(self.project)
            logging.info(f"Project directory created at: {self.project}")
        except FileExistsError:
            logging.info(f"A project exists at: {self.project}")
            if self.overwrite:
                logging.info(f"Every file is going to be overwritten in this directory!")
            else:
                logging.info(f"You used flag to prevent the project to be overwritten, so we are terminating now...")
                quit(0)
        # generating dirs

        self.logs = os.path.join(self.project, "logs")
        self.models = os.path.join(self.project, "saves")
        self.results = os.path.join(self.project, "results")
        try:
            os.mkdir(self.logs)
            logging.info(f"Logging directory created at: {self.logs}")
        except FileExistsError:
            pass

        try:
            os.mkdir(self.models)
            logging.info(f"Models directory created at: {self.models}")
        except FileExistsError:
            pass

        try:
            os.mkdir(self.results)
            logging.info(f"Results directory created at: {self.results}")
        except FileExistsError:
            pass

    def to_json(self) -> str:
        return json.dumps(
            {
                "workspace": self.workspace,
                "name": self.name,
                "project": self.project,
                "logs": self.logs,
                "models": self.models,
                "results": self.results,
                "processes": self.processes
            })

    def from_dict(self, params):
        self._workspace = params["workspace"]
        self._name = params["name"]
        self.project = params["project"]
        self.logs = params["logs"]
        self.models = params["models"]
        self.results = params["results"]
        self.processes = params["processes"]
        self._structure_generated = True

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return json.loads(self.to_json())
