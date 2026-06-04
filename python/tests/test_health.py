import pytest
import tempfile
from pathlib import Path

from sodl_weights.health import HealthMonitor, HealthStatus, ComponentHealth, HealthReport


class TestHealthMonitor:
    @pytest.fixture
    def monitor(self, tmp_path):
        return HealthMonitor(disk_path=str(tmp_path))

    def test_register_and_check(self, monitor):
        monitor.register_check("store", lambda: True)
        comp = monitor.check_component("store")
        assert comp.status == HealthStatus.HEALTHY
        assert comp.latency_ms >= 0

    def test_failed_check(self, monitor):
        monitor.register_check("bad", lambda: False)
        comp = monitor.check_component("bad")
        assert comp.status == HealthStatus.DEGRADED

    def test_error_check(self, monitor):
        monitor.register_check("crash", lambda: 1/0)
        comp = monitor.check_component("crash")
        assert comp.status == HealthStatus.UNHEALTHY
        assert "division" in comp.message

    def test_unregistered(self, monitor):
        comp = monitor.check_component("nonexistent")
        assert comp.status == HealthStatus.UNHEALTHY

    def test_deregister(self, monitor):
        monitor.register_check("temp", lambda: True)
        monitor.deregister_check("temp")
        comp = monitor.check_component("temp")
        assert comp.status == HealthStatus.UNHEALTHY

    def test_overall_healthy(self, monitor):
        monitor.register_check("a", lambda: True)
        monitor.register_check("b", lambda: True)
        report = monitor.check_health()
        assert report.status == HealthStatus.HEALTHY
        assert report.is_live
        assert report.is_ready

    def test_overall_degraded(self, monitor):
        monitor.register_check("a", lambda: True)
        monitor.register_check("b", lambda: False)
        report = monitor.check_health()
        assert report.status == HealthStatus.DEGRADED
        assert report.is_live
        assert not report.is_ready

    def test_resources(self, monitor):
        resources = monitor.check_resources()
        assert resources.disk_total_gb > 0
        assert resources.disk_pct >= 0

    def test_liveness(self, monitor):
        monitor.register_check("core", lambda: True)
        assert monitor.is_live()

    def test_readiness(self, monitor):
        monitor.register_check("core", lambda: True)
        assert monitor.is_ready()
