"""ResearchKit Python SDK"""

__version__ = "1.0.0"
__author__ = "Giland.YIN"

from .client import ResearchKitClient
from .data_processor import DataProcessor
from .models import (
    SensorType,
    SensorConfig,
    SensorData,
    SensorChannels,
    BLEDevice,
    ConnectionStatus,
)
from .exceptions import (
    RSKSDKError,
    BLEConnectionError,
    BLETimeoutError,
    SensorConfigError,
    DataProcessError,
    DeviceNotFoundError,
    CommandAckError,
)

__all__ = [
    "ResearchKitClient",
    "DataProcessor",
    "SensorType",
    "SensorConfig",
    "SensorData",
    "SensorChannels",
    "BLEDevice",
    "ConnectionStatus",
    "RSKSDKError",
    "BLEConnectionError",
    "BLETimeoutError",
    "SensorConfigError",
    "DataProcessError",
    "DeviceNotFoundError",
    "CommandAckError",
]
