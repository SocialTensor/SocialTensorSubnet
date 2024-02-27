from abc import ABC, abstractmethod


class BaseT2IModel(ABC):
    def __init__(self, *args, **kwargs):
        self.inference_function = self.load_model(*args, **kwargs)

    @abstractmethod
    def load_model(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        return self.inference_function(*args, **kwargs)
