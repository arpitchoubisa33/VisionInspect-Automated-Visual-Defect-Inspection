"""VisionInspect - Automated Visual Quality Inspection System.

A modular computer-vision pipeline that ingests surface images, segments
candidate defect regions, extracts shape/intensity/texture descriptors,
classifies each region with a from-scratch k-NN model, and produces
inspection reports.
"""

__version__ = "1.0.0"
__author__ = "VITyarthi Student"

CLASS_NAMES = ("good", "scratch", "spot", "crack")
DEFECT_CLASSES = ("scratch", "spot", "crack")
