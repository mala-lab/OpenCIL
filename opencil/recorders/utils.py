from opencil.utils import Config
from .base_recorder import BaseRecorder


def get_recorder(config: Config):
    recorders = {
        'base': BaseRecorder,
    }

    return recorders[config.recorder.name](config)
