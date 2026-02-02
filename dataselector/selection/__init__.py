"""Selection algorithms (canonical import paths).

For now, implementations still live in `src/` and are re-exported here.
This gives us a stable public API while we migrate code incrementally.
"""

from dataselector.selection.diversity_selector import DiversitySelector
from dataselector.selection.multi_criteria_facility_location import MultiCriteriaFacilityLocation
from dataselector.selection.spatial_facility_location import SpatialConstrainedFacilityLocation

__all__ = [
    "DiversitySelector",
    "MultiCriteriaFacilityLocation",
    "SpatialConstrainedFacilityLocation",
]
