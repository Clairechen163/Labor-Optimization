from pathlib import Path

import pytest
import yaml

from labor_opt.config import load_scenario

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def casino():
    return load_scenario(ROOT / "configs" / "casino.yaml")


@pytest.fixture(scope="session")
def retail():
    return load_scenario(ROOT / "configs" / "retail.yaml")


@pytest.fixture
def make_scenario(tmp_path):
    """Build a small scenario from overrides on a one-role, one-day template."""
    def _make(**over):
        cfg = {
            "name": "tiny", "industry": "test", "currency": "HKD", "seed": 0,
            "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "operating_days": ["Mon"],
            "blocks": {"all": [0, 24]},
            "roles": {"staff": {"wage": 100}},
            "demand": {"staff": {"all": 1}},
            "shift_templates": [{"name": "a", "start": 0, "length": 8}, {"name": "b", "start": 8, "length": 8},
                                {"name": "c", "start": 16, "length": 8}],
            "baseline_templates": ["a", "b", "c"],
            "workforce": {"staff": [{"count": 3, "contract": "FT"}]},
            "contracts": {"FT": {"min_hours": 0, "max_hours": 40, "max_days": 5}},
            "availability": {},
            "rules": {"min_rest_hours": 0, "overtime_threshold": 40, "overtime_premium": 1.5,
                      "max_overtime_hours": 0, "call_in_multiplier": 3.0},
        }
        for k, v in over.items():
            cfg[k] = v
        p = tmp_path / "tiny.yaml"
        p.write_text(yaml.safe_dump(cfg))
        return load_scenario(p)
    return _make
