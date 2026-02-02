"""Selection algorithms (canonical import paths).

For now, implementations still live in `src/` and are re-exported here.
This gives us a stable public API while we migrate code incrementally.
"""

from src.diversity_selector import DiversitySelector
from src.multi_criteria_facility_location import MultiCriteriaFacilityLocation
from src.spatial_facility_location import SpatialConstrainedFacilityLocation

__all__ = [
    "DiversitySelector",
    "MultiCriteriaFacilityLocation",
    "SpatialConstrainedFacilityLocation",
]
