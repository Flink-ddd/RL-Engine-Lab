import torch
import logging

# Configure infrastructure-level logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceContext:
    """
    Hardware-aware context manager for high-performance RL tasks.
    
    Provides transparent support for both AMD (ROCm/HIP) and NVIDIA (CUDA) 
    architectures to ensure backend-agnostic scaling for RL operators.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_rocm = False
        self.backend_version = "N/A"

        if self.device.type == "cuda":
            # Distinct detection for AMD HIP vs. NVIDIA CUDA
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                self.is_rocm = True
                self.backend_version = torch.version.hip
                logger.info(f"RL-Engine initialized with AMD ROCm backend (Version: {self.backend_version})")
            else:
                self.is_rocm = False
                self.backend_version = torch.version.cuda
                logger.info(f"RL-Engine initialized with NVIDIA CUDA backend (Version: {self.backend_version})")
        else:
            logger.warning("No GPU detected. RL-Engine is falling back to CPU mode.")

    def get_preferred_dtype(self):
        """
        Returns the optimal data type for the current hardware.
        AMD ROCm typically yields better performance with bfloat16 in RL workloads.
        """
        return torch.bfloat16 if self.is_rocm else torch.float16

# Global singleton for hardware context access
device_ctx = DeviceContext()