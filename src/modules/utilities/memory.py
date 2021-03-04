import os
import platform
import time
import sys
from src.modules.utilities.logging import Logger


def construct_shared_memory_name(name: str):
    if platform.system() == 'Linux':
        memory_prefix = f"{os.getuid()}_{os.getpid()}_{int(round(time.time() * 1000))}"
    elif platform.system() == 'Windows':
        memory_prefix = f"{int(round(time.time() * 1000))}"
    else:
        Logger().logger.error("The OS is not supported!")
        sys.exit(1)
    return f"{memory_prefix}-{name}"
