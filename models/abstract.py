import abc


class Model(abc.ABC):

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractmethod
    def load(self):
        pass
