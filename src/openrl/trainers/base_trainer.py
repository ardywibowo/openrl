from openrl.common import Component
from openrl.common.logging_utils import get_logger

logger = get_logger(__name__)


class Trainer(Component):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
