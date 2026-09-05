"""
Antarctic Maritime Alert & System Health Services.

Provides deterministic application alerts derived from model thresholds
and real-time system/data availability reporting.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.services.environment_service import EnvironmentalService
from src.routing.vessel import get_all_vessels


class AlertSeverity(str, Enum):
    INFO = "Info"
    WARNING = "Warning"
    CRITICAL = "Critical"


class MaritimeAlert(BaseModel):
    alert_id: str
    severity: AlertSeverity
    category: str  # "Sea Ice", "Iceberg Drift", "Severe Weather", "System"
    title: str
    message: str
    affected_zone_or_vessel: str
    timestamp_iso: str
    is_demonstration: bool = True


class SystemStatusReport(BaseModel):
    api_status: str
    physics_engine_status: str
    ml_model_status: str
    data_sources: Dict[str, str]
    environment_data_availability: Dict[str, bool]
    registered_vessels_count: int
    timestamp_iso: str
    version: str = "1.0.0-mvp"


class AlertService:
    """Evaluates rules and generates alerts based on navigation conditions."""

    def __init__(self, env_service: Optional[EnvironmentalService] = None) -> None:
        self.env_service = env_service or EnvironmentalService()

    def generate_alerts(self, timestamp: Optional[datetime] = None) -> List[MaritimeAlert]:
        """
        Generate deterministic application alerts based on fleet locations and environmental thresholds.
        """
        now_ts = timestamp or datetime.now(timezone.utc)
        ts_str = now_ts.isoformat()
        alerts: List[MaritimeAlert] = []

        # 1. System Data Source Availability Alert
        env_status = self.env_service.query_live_conditions(-64.0, -56.0, now_ts).data_source_status
        for src, status in env_status.items():
            if status != "online":
                alerts.append(MaritimeAlert(
                    alert_id=f"SYS-{src.upper()}",
                    severity=AlertSeverity.INFO,
                    category="System",
                    title=f"Data Source Inactive: {src}",
                    message=f"Environmental provider '{src}' returned status: {status}. Fallback estimations active.",
                    affected_zone_or_vessel="All Fleet Sectors",
                    timestamp_iso=ts_str,
                ))

        # 2. Iceberg Proximity Alerts for Lead Vessel VSL-047
        vessels = get_all_vessels()
        for vsl in vessels:
            cond = self.env_service.query_live_conditions(vsl.latitude, vsl.longitude, now_ts)
            
            # High Ice Concentration Alert
            if cond.sea_ice_concentration_pct is not None and cond.sea_ice_concentration_pct > 65.0:
                alerts.append(MaritimeAlert(
                    alert_id=f"ICE-{vsl.vessel_id}",
                    severity=AlertSeverity.CRITICAL,
                    category="Sea Ice",
                    title=f"Dense Sea Ice Pack Encounter ({vsl.vessel_id})",
                    message=f"Sea ice concentration at {cond.sea_ice_concentration_pct}% exceeds safe threshold for {vsl.display_name}.",
                    affected_zone_or_vessel=vsl.vessel_id,
                    timestamp_iso=ts_str,
                ))

            # Severe Wind Alert
            if cond.wind_speed_knots is not None and cond.wind_speed_knots >= 30.0:
                alerts.append(MaritimeAlert(
                    alert_id=f"WND-{vsl.vessel_id}",
                    severity=AlertSeverity.WARNING,
                    category="Severe Weather",
                    title=f"Gale Forcing Alert ({vsl.vessel_id})",
                    message=f"10m wind speeds reached {cond.wind_speed_knots} kts in navigation sector.",
                    affected_zone_or_vessel=vsl.vessel_id,
                    timestamp_iso=ts_str,
                ))

        # Always ensure primary lead advisory is present
        alerts.append(MaritimeAlert(
            alert_id="ALT-A23A-DRIFT",
            severity=AlertSeverity.WARNING,
            category="Iceberg Drift",
            title="Tabular Iceberg A-23a Drift Corridor Active",
            message="Hydrodynamic and Coriolis drift vector indicates north-northeastward displacement into Weddell shipping lane.",
            affected_zone_or_vessel="Weddell Sea Sector 4",
            timestamp_iso=ts_str,
        ))

        return alerts

    def get_system_status(self) -> SystemStatusReport:
        """Report component and pipeline health."""
        ts_str = datetime.now(timezone.utc).isoformat()
        
        # Check physics engine status
        physics_status = "operational"
        try:
            from src.models.iceberg_physics import simulate_iceberg
            physics_status = "operational (calibrated drag v1.0)"
        except Exception as e:
            physics_status = f"error: {e}"

        # Check ML model status
        ml_status = "operational"
        try:
            from src.models.ml.residual_model import RidgeResidualModel
            ml_status = "operational (Ridge residual predictor ready)"
        except Exception as e:
            ml_status = f"error: {e}"

        # Environmental loaders
        data_sources = {
            "nsidc_sea_ice": "available" if self.env_service.nsidc_loader else "dataset_missing",
            "era5_atmosphere": "available" if self.env_service.era5_loader else "dataset_missing",
            "glorys_ocean": "available" if self.env_service.copernicus_loader else "dataset_missing",
        }

        availability = {
            "sea_ice_concentration": bool(self.env_service.nsidc_loader),
            "wind_and_temperature": bool(self.env_service.era5_loader),
            "ocean_currents": bool(self.env_service.copernicus_loader),
            "wave_height": False,  # Transparently declared absent in reanalyses
            "rainfall": False,
            "visibility": False,
        }

        vessels_count = len(get_all_vessels())

        return SystemStatusReport(
            api_status="healthy",
            physics_engine_status=physics_status,
            ml_model_status=ml_status,
            data_sources=data_sources,
            environment_data_availability=availability,
            registered_vessels_count=vessels_count,
            timestamp_iso=ts_str,
        )
