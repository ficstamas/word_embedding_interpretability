class MemoryInfo:
    """
    A Wrapper to store SharedMemory object properties
    """
    def __init__(self):
        self.name = ""
        self.shape = None
        self.dtype = None

    def __str__(self):
        s = f"MemoryInfo[\nname={self.name},\nshape={self.shape}\n]"
        return s
