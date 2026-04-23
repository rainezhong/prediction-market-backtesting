# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software distributed under the
#  License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
# -------------------------------------------------------------------------------------------------
#  Derived from NautilusTrader prediction-market example code.
#  See the repository NOTICE file for provenance and licensing scope.
#

"""Tests for custom Nautilus data types."""

from __future__ import annotations

from decimal import Decimal

import pytest
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import CustomData, DataType
from nautilus_trader.model.identifiers import InstrumentId

from prediction_market_extensions.data.custom_types import (
    BasketballContextData,
    F1TelemetryData,
    basketball_data_type,
    f1_data_type,
)
from strategies.basketball_context_demo import (
    BasketballContextDemo,
    BasketballContextDemoConfig,
)


# ---------------------------------------------------------------------------
# BasketballContextData
# ---------------------------------------------------------------------------


class TestBasketballContextData:
    """Construction and derived property tests for BasketballContextData."""

    def _make(self, **overrides):
        defaults = dict(
            home_team="LAL",
            away_team="BOS",
            home_score=88,
            away_score=85,
            period=3,
            game_clock_secs=342.0,
            is_halftime=False,
            is_final=False,
            ts_event=1000,
            ts_init=2000,
        )
        defaults.update(overrides)
        return BasketballContextData(**defaults)

    def test_construction(self):
        ctx = self._make()
        assert ctx.home_team == "LAL"
        assert ctx.away_team == "BOS"
        assert ctx.home_score == 88
        assert ctx.away_score == 85
        assert ctx.period == 3
        assert ctx.game_clock_secs == 342.0
        assert ctx.is_halftime is False
        assert ctx.is_final is False

    def test_is_data_subclass(self):
        ctx = self._make()
        assert isinstance(ctx, Data)

    def test_ts_event_ts_init(self):
        ctx = self._make(ts_event=111, ts_init=222)
        assert ctx.ts_event == 111
        assert ctx.ts_init == 222

    def test_ts_defaults_zero(self):
        ctx = BasketballContextData(
            home_team="A",
            away_team="B",
            home_score=0,
            away_score=0,
            period=1,
            game_clock_secs=720.0,
        )
        assert ctx.ts_event == 0
        assert ctx.ts_init == 0

    def test_total_score(self):
        ctx = self._make(home_score=50, away_score=45)
        assert ctx.total_score == 95

    def test_score_diff_home_leading(self):
        ctx = self._make(home_score=90, away_score=80)
        assert ctx.score_diff == 10

    def test_score_diff_away_leading(self):
        ctx = self._make(home_score=70, away_score=85)
        assert ctx.score_diff == -15

    def test_score_diff_tied(self):
        ctx = self._make(home_score=60, away_score=60)
        assert ctx.score_diff == 0

    def test_is_close_game_true(self):
        ctx = self._make(home_score=80, away_score=76)
        assert ctx.is_close_game is True

    def test_is_close_game_boundary(self):
        ctx = self._make(home_score=80, away_score=75)
        assert ctx.is_close_game is True

    def test_is_close_game_false(self):
        ctx = self._make(home_score=80, away_score=74)
        assert ctx.is_close_game is False

    def test_is_close_game_tied(self):
        ctx = self._make(home_score=60, away_score=60)
        assert ctx.is_close_game is True

    def test_period_display_normal(self):
        ctx = self._make(period=3, game_clock_secs=342.0)
        assert ctx.period_display == "Q3 5:42"

    def test_period_display_zero_seconds(self):
        ctx = self._make(period=4, game_clock_secs=0.0)
        assert ctx.period_display == "Q4 0:00"

    def test_period_display_halftime(self):
        ctx = self._make(is_halftime=True)
        assert ctx.period_display == "HALF"

    def test_period_display_final(self):
        ctx = self._make(is_final=True)
        assert ctx.period_display == "FINAL"

    def test_period_display_final_takes_precedence(self):
        ctx = self._make(is_final=True, is_halftime=True)
        assert ctx.period_display == "FINAL"

    def test_period_zero_edge(self):
        ctx = self._make(period=0, game_clock_secs=0.0)
        assert ctx.period_display == "Q0 0:00"

    def test_repr(self):
        ctx = self._make(home_team="LAL", away_team="BOS", home_score=88, away_score=85)
        r = repr(ctx)
        assert "LAL" in r
        assert "BOS" in r
        assert "88" in r
        assert "85" in r


# ---------------------------------------------------------------------------
# F1TelemetryData
# ---------------------------------------------------------------------------


class TestF1TelemetryData:
    """Construction and derived property tests for F1TelemetryData."""

    def _make(self, **overrides):
        defaults = dict(
            driver="VER",
            position=1,
            lap=40,
            total_laps=57,
            gap_to_leader_secs=0.0,
            tire_compound="medium",
            pit_count=1,
            is_pit_window=False,
            ts_event=5000,
            ts_init=6000,
        )
        defaults.update(overrides)
        return F1TelemetryData(**defaults)

    def test_construction(self):
        td = self._make()
        assert td.driver == "VER"
        assert td.position == 1
        assert td.lap == 40
        assert td.total_laps == 57
        assert td.gap_to_leader_secs == 0.0
        assert td.tire_compound == "medium"
        assert td.pit_count == 1
        assert td.is_pit_window is False

    def test_is_data_subclass(self):
        td = self._make()
        assert isinstance(td, Data)

    def test_ts_event_ts_init(self):
        td = self._make(ts_event=999, ts_init=888)
        assert td.ts_event == 999
        assert td.ts_init == 888

    def test_ts_defaults_zero(self):
        td = F1TelemetryData(
            driver="HAM",
            position=2,
            lap=1,
            total_laps=50,
            gap_to_leader_secs=1.5,
            tire_compound="soft",
            pit_count=0,
        )
        assert td.ts_event == 0
        assert td.ts_init == 0

    def test_laps_remaining(self):
        td = self._make(lap=40, total_laps=57)
        assert td.laps_remaining == 17

    def test_laps_remaining_at_finish(self):
        td = self._make(lap=57, total_laps=57)
        assert td.laps_remaining == 0

    def test_laps_remaining_never_negative(self):
        td = self._make(lap=60, total_laps=57)
        assert td.laps_remaining == 0

    def test_race_progress_middle(self):
        td = self._make(lap=28, total_laps=56)
        assert td.race_progress == pytest.approx(0.5)

    def test_race_progress_start(self):
        td = self._make(lap=0, total_laps=57)
        assert td.race_progress == pytest.approx(0.0)

    def test_race_progress_end(self):
        td = self._make(lap=57, total_laps=57)
        assert td.race_progress == pytest.approx(1.0)

    def test_race_progress_zero_total_laps(self):
        td = self._make(lap=5, total_laps=0)
        assert td.race_progress == 0.0

    def test_race_progress_capped_at_one(self):
        td = self._make(lap=60, total_laps=57)
        assert td.race_progress == 1.0

    def test_is_late_race_true(self):
        td = self._make(lap=44, total_laps=57)
        assert td.is_late_race is True

    def test_is_late_race_false(self):
        td = self._make(lap=42, total_laps=57)
        assert td.is_late_race is False

    def test_is_late_race_boundary(self):
        # Exactly 75% should be False (> 0.75 required)
        td = self._make(lap=75, total_laps=100)
        assert td.is_late_race is False

    def test_is_late_race_zero_laps(self):
        td = self._make(lap=0, total_laps=0)
        assert td.is_late_race is False

    def test_repr(self):
        td = self._make(driver="HAM", position=3, lap=20, total_laps=57)
        r = repr(td)
        assert "HAM" in r
        assert "P3" in r
        assert "20/57" in r


# ---------------------------------------------------------------------------
# DataType helpers
# ---------------------------------------------------------------------------


class TestDataTypeHelpers:
    """Test the basketball_data_type and f1_data_type helper functions."""

    def test_basketball_data_type_no_event(self):
        dt = basketball_data_type()
        assert dt.type is BasketballContextData
        assert dt.metadata == {"sport": "basketball"}

    def test_basketball_data_type_with_event(self):
        dt = basketball_data_type(event_id="KXNBA-LAL-BOS")
        assert dt.type is BasketballContextData
        assert dt.metadata == {"sport": "basketball", "event_id": "KXNBA-LAL-BOS"}

    def test_f1_data_type_default_session(self):
        dt = f1_data_type()
        assert dt.type is F1TelemetryData
        assert dt.metadata == {"sport": "f1", "session": "race"}

    def test_f1_data_type_qualifying(self):
        dt = f1_data_type(session="qualifying")
        assert dt.metadata["session"] == "qualifying"

    def test_data_type_is_nautilus_datatype(self):
        dt = basketball_data_type()
        assert isinstance(dt, DataType)

    def test_f1_data_type_is_nautilus_datatype(self):
        dt = f1_data_type()
        assert isinstance(dt, DataType)


# ---------------------------------------------------------------------------
# CustomData wrapping
# ---------------------------------------------------------------------------


class TestCustomDataWrapping:
    """Test that custom data works inside Nautilus CustomData envelope."""

    def test_basketball_in_custom_data(self):
        ctx = BasketballContextData(
            home_team="LAL",
            away_team="BOS",
            home_score=100,
            away_score=95,
            period=4,
            game_clock_secs=60.0,
            ts_event=123,
            ts_init=456,
        )
        dt = basketball_data_type()
        cd = CustomData(data_type=dt, data=ctx)
        assert isinstance(cd.data, BasketballContextData)
        assert cd.data.total_score == 195
        assert cd.data.ts_event == 123

    def test_f1_in_custom_data(self):
        tel = F1TelemetryData(
            driver="VER",
            position=1,
            lap=50,
            total_laps=57,
            gap_to_leader_secs=0.0,
            tire_compound="hard",
            pit_count=2,
            ts_event=789,
            ts_init=101,
        )
        dt = f1_data_type()
        cd = CustomData(data_type=dt, data=tel)
        assert isinstance(cd.data, F1TelemetryData)
        assert cd.data.laps_remaining == 7
        assert cd.data.ts_init == 101

    def test_isinstance_check_pattern(self):
        """Verify the pattern strategies use in on_data()."""
        ctx = BasketballContextData(
            home_team="A",
            away_team="B",
            home_score=0,
            away_score=0,
            period=1,
            game_clock_secs=720.0,
        )
        dt = basketball_data_type()
        cd = CustomData(data_type=dt, data=ctx)
        # This is how strategies check:
        assert isinstance(cd, CustomData)
        assert isinstance(cd.data, BasketballContextData)


# ---------------------------------------------------------------------------
# Demo strategy config
# ---------------------------------------------------------------------------


class TestBasketballContextDemoConfig:
    """Test that the demo strategy config can be created."""

    def test_config_creation(self):
        config = BasketballContextDemoConfig(
            instrument_id=InstrumentId.from_str("KXNBA-LAL-BOS-YES.KALSHI"),
            trade_size=Decimal(5),
            event_id="KXNBA-LAL-BOS",
        )
        assert str(config.instrument_id) == "KXNBA-LAL-BOS-YES.KALSHI"
        assert config.trade_size == Decimal(5)
        assert config.event_id == "KXNBA-LAL-BOS"

    def test_config_defaults(self):
        config = BasketballContextDemoConfig(
            instrument_id=InstrumentId.from_str("TEST.VENUE"),
        )
        assert config.trade_size == Decimal(1)
        assert config.event_id == ""


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------


class TestPackageImport:
    """Verify the data package re-exports types correctly."""

    def test_import_from_package(self):
        from prediction_market_extensions.data import (
            BasketballContextData,
            F1TelemetryData,
            basketball_data_type,
            f1_data_type,
        )

        assert BasketballContextData is not None
        assert F1TelemetryData is not None
        assert callable(basketball_data_type)
        assert callable(f1_data_type)
