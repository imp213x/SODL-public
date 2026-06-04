"""Health Monitor — component health checks and system resource monitoring.

Provides liveness/readiness probes and resource usage alerts.

Example
-------
>>> monitor = HealthMonitor()
>>> monitor.register_check("store", lambda: store_root.exists())
>>> status = monitor.check_health()
"""

from __future__ import annotations

import os
import time
import shutil
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus = HealthStatus.HEALTHY
    message: str = ""
    latency_ms: float = 0.0
    last_checked: float = 0.0


@dataclass
class SystemResources:
    """Snapshot of system resource usage."""
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_pct: float = 0.0
    process_memory_mb: float = 0.0

    @property
    def disk_critical(self) -> bool:
        return self.disk_pct > 95

    @property
    def disk_warning(self) -> bool:
        return self.disk_pct > 85


@dataclass
class HealthReport:
    """Overall health report."""
    status: HealthStatus = HealthStatus.HEALTHY
    components: list[ComponentHealth] = field(default_factory=list)
    resources: Optional[SystemResources] = None
    timestamp: float = 0.0

    @property
    def is_live(self) -> bool:
        return self.status != HealthStatus.UNHEALTHY

    @property
    def is_ready(self) -> bool:
        return self.status == HealthStatus.HEALTHY


class HealthMonitor:
    """System health monitor with component checks and resource tracking.

    Parameters
    ----------
    disk_path : str or Path, optional
        Path to monitor disk usage on.
    disk_warn_pct : float
        Disk usage percentage to trigger warning.
    disk_crit_pct : float
        Disk usage percentage to trigger critical alert.
    """

    def __init__(
        self,
        disk_path: Optional[str | Path] = None,
        disk_warn_pct: float = 85.0,
        disk_crit_pct: float = 95.0,
    ) -> None:
        self._checks: dict[str, Callable[[], bool]] = {}
        self._disk_path = Path(disk_path) if disk_path else None
        self._disk_warn_pct = disk_warn_pct
        self._disk_crit_pct = disk_crit_pct
        self._last_report: Optional[HealthReport] = None

    def register_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        """Register a health check function. Should return True if healthy."""
        self._checks[name] = check_fn

    def deregister_check(self, name: str) -> None:
        """Remove a health check."""
        self._checks.pop(name, None)

    def check_component(self, name: str) -> ComponentHealth:
        """Run a single component health check."""
        check_fn = self._checks.get(name)
        if check_fn is None:
            return ComponentHealth(name=name, status=HealthStatus.UNHEALTHY, message="Not registered")

        start = time.perf_counter()
        try:
            result = check_fn()
            latency = (time.perf_counter() - start) * 1000
            if result:
                return ComponentHealth(
                    name=name, status=HealthStatus.HEALTHY,
                    latency_ms=round(latency, 2), last_checked=time.time(),
                )
            else:
                return ComponentHealth(
                    name=name, status=HealthStatus.DEGRADED,
                    message="Check returned False",
                    latency_ms=round(latency, 2), last_checked=time.time(),
                )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name=name, status=HealthStatus.UNHEALTHY,
                message=str(e), latency_ms=round(latency, 2),
                last_checked=time.time(),
            )

    def check_resources(self) -> SystemResources:
        """Check system resource usage."""
        resources = SystemResources()

        # Disk
        if self._disk_path and self._disk_path.exists():
            usage = shutil.disk_usage(str(self._disk_path))
            resources.disk_total_gb = round(usage.total / (1024**3), 2)
            resources.disk_used_gb = round(usage.used / (1024**3), 2)
            resources.disk_free_gb = round(usage.free / (1024**3), 2)
            resources.disk_pct = round((usage.used / usage.total) * 100, 1) if usage.total > 0 else 0

        # Process memory
        try:
            import psutil
            proc = psutil.Process()
            resources.process_memory_mb = round(proc.memory_info().rss / (1024**2), 1)
        except (ImportError, Exception):
            pass

        return resources

    def check_health(self) -> HealthReport:
        """Run all health checks and return a comprehensive report."""
        report = HealthReport(timestamp=time.time())

        # Component checks
        for name in self._checks:
            comp = self.check_component(name)
            report.components.append(comp)

        # Resources
        report.resources = self.check_resources()

        # Overall status
        statuses = [c.status for c in report.components]
        if report.resources and report.resources.disk_critical:
            report.status = HealthStatus.UNHEALTHY
        elif HealthStatus.UNHEALTHY in statuses:
            report.status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses or (report.resources and report.resources.disk_warning):
            report.status = HealthStatus.DEGRADED
        else:
            report.status = HealthStatus.HEALTHY

        self._last_report = report
        return report

    def is_live(self) -> bool:
        """Liveness probe: True if system is not fatal."""
        report = self.check_health()
        return report.is_live

    def is_ready(self) -> bool:
        """Readiness probe: True if system is fully healthy."""
        report = self.check_health()
        return report.is_ready

    @property
    def last_report(self) -> Optional[HealthReport]:
        return self._last_report
