"""Project-wide exception hierarchy.

A single base class lets the CLI catch every expected failure in one place
while still allowing callers to react to specific problems.
"""


class VisionInspectError(Exception):
    """Base class for all errors raised by VisionInspect."""


class ConfigError(VisionInspectError):
    """Raised when a configuration value is missing or invalid."""


class ImageLoadError(VisionInspectError):
    """Raised when an image file cannot be read or is malformed."""


class DatasetError(VisionInspectError):
    """Raised when a dataset is empty, unbalanced beyond repair, or missing."""


class ModelError(VisionInspectError):
    """Raised when a model is used before training or is given bad input."""
