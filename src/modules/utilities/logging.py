import logging
import os
from .metaclasses import Singleton
import tqdm

class Logger(metaclass=Singleton):
    def __init__(self):
        self.logger = logging.getLogger("default")

    def setup(self, project: str):
        stream_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s',
                                             datefmt='%d-%b-%y %H:%M:%S')
        file_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s | [%(module)s|%(funcName)s] - %(message)s',
                                           datefmt='%d-%b-%y %H:%M:%S')

        file_handler = logging.FileHandler(os.path.join(project, "logs/debug.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        error_file_handler = logging.FileHandler(os.path.join(project, "logs/error.log"))
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(file_formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(stream_formatter)

        tqdm_handler = TqdmLoggingHandler(logging.INFO)
        tqdm_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(error_file_handler)
        # self.logger.addHandler(tqdm_handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Logging Handlers are initiated!")


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
