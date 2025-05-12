# Register all processors here

from .base import ProcessorRegistry, Processor
import pkgutil
import importlib


def register_all_processors():
    for _, module_name, _ in pkgutil.iter_modules([__path__[0]]):
        module = importlib.import_module(f"{__name__}.{module_name}")
        for attr in dir(module):
            cls = getattr(module, attr)
            if (
                isinstance(cls, type)
                and issubclass(cls, Processor)
                and cls is not Processor
            ):
                ProcessorRegistry.register(cls)


register_all_processors()
