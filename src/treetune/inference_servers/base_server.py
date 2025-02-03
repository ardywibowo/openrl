from treetune.common import Component
from treetune.common.logging_utils import get_logger

logger = get_logger(__name__)

class InferenceServer(Component):
    def __init__(
        self,
        swap_space: int = 8,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        assert isinstance(gpu_memory_utilization, float)
        assert 0.0 < gpu_memory_utilization <= 1.0
        
        self.swap_space = swap_space
        self.gpu_memory_utilization = gpu_memory_utilization

    def start_server(self, *args, **kwargs) -> str:
        raise NotImplementedError

    def stop_server(self):
        raise NotImplementedError
