"""Selection algorithms public surface.

Exports are resolved lazily to avoid importing optional heavy dependencies at
package import time.
"""

from __future__ import annotations

__all__ = [
    "DiversitySelector",
    "MultiCriteriaFacilityLocation",
    "SpatialConstrainedFacilityLocation",
]


def __getattr__(name: str):
    if name == "DiversitySelector":
        from dataselector.selection.diversity_selector import DiversitySelector

        return DiversitySelector
    if name == "MultiCriteriaFacilityLocation":
        from dataselector.selection.multi_criteria_facility_location import (
            MultiCriteriaFacilityLocation,
        )

        return MultiCriteriaFacilityLocation
    if name == "SpatialConstrainedFacilityLocation":
        from dataselector.selection.spatial_facility_location import (
            SpatialConstrainedFacilityLocation,
        )

        return SpatialConstrainedFacilityLocation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
