"""Scenario loading: YAML config -> requirements, shift templates and a workforce."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HOURS_PER_WEEK = 168
WEEKEND = ("Sat", "Sun")


@dataclass(frozen=True)
class ShiftTemplate:
    name: str
    start: int          # hour of day, 0-23
    length: int         # hours
    premium: float = 1.0

    def covered_hours(self, day: int) -> list[int]:
        """Absolute week hours (0-167) covered when the shift starts on `day`. Wraps at week end."""
        t0 = day * 24 + self.start
        return [(t0 + i) % HOURS_PER_WEEK for i in range(self.length)]

    @property
    def crosses_midnight(self) -> bool:
        return self.start + self.length > 24 or self.start < 6


@dataclass
class Role:
    name: str
    wage: float
    max_on_duty: int | None = None


@dataclass
class Employee:
    id: str
    primary_role: str
    skills: list[str]
    contract: str
    wage: float
    min_hours: float
    max_hours: float
    max_days: int
    allowed_lengths: list[int] | None = None
    no_nights: bool = False
    days_off: set[int] = field(default_factory=set)

    def can_work(self, day: int, tpl: ShiftTemplate) -> bool:
        if day in self.days_off:
            return False
        if self.no_nights and tpl.crosses_midnight:
            return False
        if self.allowed_lengths and tpl.length not in self.allowed_lengths:
            return False
        return True


@dataclass
class Scenario:
    name: str
    industry: str
    currency: str
    days: list[str]
    operating_days: list[str]
    roles: dict[str, Role]
    templates: list[ShiftTemplate]
    baseline_templates: list[str]
    requirements: pd.DataFrame        # columns: role, day, hour, t, required
    employees: list[Employee]
    rules: dict
    raw: dict

    def template(self, name: str) -> ShiftTemplate:
        return next(t for t in self.templates if t.name == name)

    def requirement_matrix(self) -> pd.DataFrame:
        """Role x week-hour matrix of required staff."""
        return self.requirements.pivot(index="role", columns="t", values="required").fillna(0)

    def with_requirements(self, req: pd.DataFrame) -> "Scenario":
        """Copy of the scenario with some roles' requirements replaced (e.g. by a forecast)."""
        keep = self.requirements[~self.requirements.role.isin(req.role.unique())]
        new = pd.concat([keep, req[keep.columns]], ignore_index=True)
        return Scenario(**{**self.__dict__, "requirements": new.sort_values(["role", "t"]).reset_index(drop=True)})


def _round_staff(x: float) -> int:
    return int(math.floor(x + 0.5))


def build_requirements(cfg: dict) -> pd.DataFrame:
    days = cfg["days"]
    operating = set(cfg.get("operating_days", days))
    blocks = cfg["blocks"]
    peak_days = set(cfg.get("peak_days", []))
    mult = cfg.get("peak_multiplier", {}) or {}
    override = cfg.get("peak_override", {}) or {}

    rows = []
    for role, per_block in cfg["demand"].items():
        for d, day in enumerate(days):
            for h in range(24):
                req = 0
                if day in operating:
                    for block, (b0, b1) in blocks.items():
                        if b0 <= h < b1:
                            if day in peak_days and role in override:
                                req = override[role].get(block, per_block.get(block, 0))
                            else:
                                req = per_block.get(block, 0)
                                if day in peak_days and role in mult:
                                    req = _round_staff(req * mult[role])
                rows.append({"role": role, "day": d, "hour": h, "t": d * 24 + h, "required": int(req)})
    return pd.DataFrame(rows)


def build_workforce(cfg: dict) -> list[Employee]:
    rng = np.random.default_rng(cfg.get("seed", 0))
    roles = cfg["roles"]
    contracts = cfg["contracts"]
    avail = cfg.get("availability", {})
    n_days = len(cfg["days"])
    operating_idx = [i for i, d in enumerate(cfg["days"]) if d in cfg.get("operating_days", cfg["days"])]

    employees: list[Employee] = []
    for role, groups in cfg["workforce"].items():
        prefix = role[:3].upper()
        n = 0
        for g in groups:
            c = contracts[g["contract"]]
            for _ in range(g["count"]):
                n += 1
                employees.append(Employee(
                    id=f"{prefix}-{n:02d}",
                    primary_role=role,
                    skills=[role],
                    contract=g["contract"],
                    wage=float(roles[role]["wage"]),
                    min_hours=float(c["min_hours"]),
                    max_hours=float(c["max_hours"]),
                    max_days=int(c["max_days"]),
                    allowed_lengths=c.get("allowed_lengths"),
                ))

    # Cross-training: the first `share` of the source role (full-timers first) gains the target skill.
    for ct in cfg.get("cross_training", []) or []:
        pool = sorted([e for e in employees if e.primary_role == ct["from"]], key=lambda e: e.contract != "FT")
        k = int(math.ceil(ct["share"] * len(pool)))
        for e in pool[:k]:
            if ct["to"] not in e.skills:
                e.skills.append(ct["to"])

    # Availability restrictions, drawn with a fixed seed so runs are reproducible.
    for e in employees:
        if rng.random() < avail.get("share_no_nights", 0.0):
            e.no_nights = True
        if rng.random() < avail.get("share_with_day_off_request", 0.0):
            choices = operating_idx if e.contract == "FT" else list(range(n_days))
            e.days_off = {int(rng.choice(choices))}
    return employees


def load_scenario(path: str | Path) -> Scenario:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    roles = {r: Role(r, float(v["wage"]), v.get("max_on_duty")) for r, v in cfg["roles"].items()}
    templates = [ShiftTemplate(t["name"], int(t["start"]), int(t["length"]), float(t.get("premium", 1.0)))
                 for t in cfg["shift_templates"]]
    return Scenario(
        name=cfg["name"],
        industry=cfg.get("industry", ""),
        currency=cfg.get("currency", ""),
        days=cfg["days"],
        operating_days=cfg.get("operating_days", cfg["days"]),
        roles=roles,
        templates=templates,
        baseline_templates=cfg.get("baseline_templates", [t.name for t in templates]),
        requirements=build_requirements(cfg),
        employees=build_workforce(cfg),
        rules=cfg.get("rules", {}),
        raw=cfg,
    )


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def scenario_path(name: str) -> Path:
    """Resolve 'casino' -> configs/casino.yaml, or pass a path through."""
    p = Path(name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    return CONFIG_DIR / f"{name}.yaml"
