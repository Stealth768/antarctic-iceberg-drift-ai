"""
Antarctic Maritime Vessel and Fleet Models.

Provides standardized demonstration vessel entities operating in Antarctic waters.
Note: These states are simulated demonstration states for navigation decision support
and route planning validation, not live satellite AIS tracks.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class VesselStatus(str, Enum):
    ACTIVE = "Active"
    ANCHORED = "Anchored"
    IN_TRANSIT = "In-Transit"


class Vessel(BaseModel):
    """Antarctic vessel representation."""
    vessel_id: str = Field(..., description="Unique vessel callsign/identifier, e.g. VSL-047")
    display_name: str = Field(..., description="Human-readable vessel name")
    polar_ice_class: str = Field(..., description="Polar ice class rating (e.g. PC3, PC5, PC7, Open Water)")
    latitude: float = Field(..., ge=-90.0, le=-50.0, description="Current latitude (Antarctic domain)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Current longitude (degrees WGS84)")
    heading: float = Field(..., ge=0.0, le=360.0, description="Vessel heading in degrees from True North")
    speed: float = Field(..., ge=0.0, le=40.0, description="Current ground speed in knots")
    destination: str = Field(..., description="Target waypoint, polar station, or operational zone")
    fuel_consumption_rate: float = Field(..., ge=0.0, description="Nominal fuel burn rate in metric tons per day")
    status: VesselStatus = Field(..., description="Operational status: Active, Anchored, or In-Transit")
    is_demonstration: bool = Field(True, description="Indicates simulated demonstration state")
    notes: Optional[str] = Field("Simulated Antarctic demonstration vessel state", description="Operational notes")


# Canonical roster of 8 demonstration vessels
DEMO_FLEET: Dict[str, Vessel] = {
    "VSL-047": Vessel(
        vessel_id="VSL-047",
        display_name="R/V Polar Pioneer (Lead)",
        polar_ice_class="PC3",
        latitude=-64.25,
        longitude=-56.75,
        heading=48.0,
        speed=12.2,
        destination="Rothera Research Station",
        fuel_consumption_rate=18.5,
        status=VesselStatus.IN_TRANSIT,
        notes="Lead expedition vessel - primary routing target in Weddell Sea corridor"
    ),
    "VSL-118": Vessel(
        vessel_id="VSL-118",
        display_name="M/V Ice Maiden",
        polar_ice_class="PC5",
        latitude=-63.80,
        longitude=-57.20,
        heading=92.0,
        speed=10.4,
        destination="Esperanza Base",
        fuel_consumption_rate=15.0,
        status=VesselStatus.IN_TRANSIT,
        notes="Supply transport bound for Antarctic Peninsula"
    ),
    "VSL-302": Vessel(
        vessel_id="VSL-302",
        display_name="R/V Kronos Explorer",
        polar_ice_class="PC7",
        latitude=-65.10,
        longitude=-60.50,
        heading=180.0,
        speed=0.0,
        destination="Palmer Station",
        fuel_consumption_rate=3.8,
        status=VesselStatus.ANCHORED,
        notes="Anchored for oceanographic station sampling"
    ),
    "VSL-409": Vessel(
        vessel_id="VSL-409",
        display_name="M/V Aurora Australis II",
        polar_ice_class="PC3",
        latitude=-62.50,
        longitude=-54.10,
        heading=215.0,
        speed=14.1,
        destination="King George Island",
        fuel_consumption_rate=22.0,
        status=VesselStatus.ACTIVE,
        notes="Icebreaker escort on standby"
    ),
    "VSL-088": Vessel(
        vessel_id="VSL-088",
        display_name="R/V Endurance Pride",
        polar_ice_class="PC5",
        latitude=-66.05,
        longitude=-58.30,
        heading=335.0,
        speed=8.6,
        destination="Weddell Sea Deep Sounding",
        fuel_consumption_rate=16.2,
        status=VesselStatus.IN_TRANSIT,
        notes="Hydrographic survey vessel"
    ),
    "VSL-221": Vessel(
        vessel_id="VSL-221",
        display_name="S/V Terra Nova",
        polar_ice_class="Open Water",
        latitude=-61.20,
        longitude=-52.90,
        heading=125.0,
        speed=9.0,
        destination="South Orkney Islands",
        fuel_consumption_rate=9.5,
        status=VesselStatus.ACTIVE,
        notes="Sub-antarctic meteorological research vessel"
    ),
    "VSL-154": Vessel(
        vessel_id="VSL-154",
        display_name="M/V Polar Sentinel",
        polar_ice_class="PC7",
        latitude=-64.80,
        longitude=-55.15,
        heading=0.0,
        speed=0.0,
        destination="Marambio Base Logistics Hub",
        fuel_consumption_rate=4.1,
        status=VesselStatus.ANCHORED,
        notes="Anchored off Seymour Island awaiting cargo clearance"
    ),
    "VSL-021": Vessel(
        vessel_id="VSL-021",
        display_name="R/V Maitri Support",
        polar_ice_class="PC5",
        latitude=-63.15,
        longitude=-59.05,
        heading=275.0,
        speed=11.5,
        destination="Bharati Station Corridor",
        fuel_consumption_rate=17.0,
        status=VesselStatus.IN_TRANSIT,
        notes="Indian Antarctic Programme support convoy"
    ),
}


def get_all_vessels() -> List[Vessel]:
    """Retrieve list of all registered demonstration vessels."""
    return list(DEMO_FLEET.values())


def get_vessel_by_id(vessel_id: str) -> Optional[Vessel]:
    """Look up vessel by identifier (case-insensitive)."""
    norm_id = vessel_id.strip().upper()
    return DEMO_FLEET.get(norm_id)
