import math

import numpy as np
import pytest

from labor_opt.demand import (erlang_c, erlang_staff, service_level, table_game_dealers,
                              theoretical_win_per_hour, workload_staff)


def test_workload():
    r = workload_staff(100_000, 25, 160, 0.9)
    assert r["workers"] == 28
    assert r["exact"] == pytest.approx(27.777, rel=1e-3)


def _erlang_c_direct(n, a):
    top = a ** n / math.factorial(n) * n / (n - a)
    return top / (sum(a ** k / math.factorial(k) for k in range(n)) + top)


@pytest.mark.parametrize("n,a", [(11, 10.0), (9, 8.33), (20, 15.5), (3, 1.2)])
def test_erlang_c_matches_closed_form(n, a):
    assert erlang_c(n, a) == pytest.approx(_erlang_c_direct(n, a), rel=1e-10)


def test_service_level_increases_with_staff():
    sls = [service_level(n, 50, 10, 2) for n in range(9, 16)]
    assert all(b > a for a, b in zip(sls, sls[1:]))


def test_erlang_staff_retail_case():
    r = erlang_staff(50, 10, 2, 0.80)
    assert r["offered_load"] == pytest.approx(50 * 10 / 60)
    assert r["servers"] == 11
    assert service_level(10, 50, 10, 2) < 0.80 <= r["service_level"]
    assert erlang_staff(50, 10, 2, 0.95, shrinkage=0.2)["scheduled"] >= 13 / 0.8


def test_table_game():
    r = table_game_dealers(120)
    assert r == {"tables": 22, "dealers": 28}
    arr = table_game_dealers(np.array([0, 5.6, 5.7]), min_tables=2, max_tables=8)
    assert list(arr["tables"]) == [2, 2, 2] and list(arr["dealers"]) == [3, 3, 3]
    assert table_game_dealers(500, max_tables=8)["tables"] == 8


def test_theoretical_win():
    assert theoretical_win_per_hour(60, 5, 500, 0.01) == pytest.approx(1500)
