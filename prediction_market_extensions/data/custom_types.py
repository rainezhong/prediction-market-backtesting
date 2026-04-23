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

"""
Custom Nautilus ``Data`` types for sports and event context.

Strategies subscribe via ``subscribe_data(basketball_data_type(...))`` and
receive instances through ``on_data()`` wrapped in ``CustomData``.
"""

from __future__ import annotations

from nautilus_trader.core.data import Data
from nautilus_trader.model.data import DataType


# ---------------------------------------------------------------------------
# Basketball
# ---------------------------------------------------------------------------


class BasketballContextData(Data):
    """Live game context for basketball (NBA / NCAAB) markets."""

    def __init__(
        self,
        *,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        period: int,
        game_clock_secs: float,
        is_halftime: bool = False,
        is_final: bool = False,
        ts_event: int = 0,
        ts_init: int = 0,
    ) -> None:
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = home_score
        self.away_score = away_score
        self.period = period
        self.game_clock_secs = game_clock_secs
        self.is_halftime = is_halftime
        self.is_final = is_final
        self._ts_event = ts_event
        self._ts_init = ts_init

    # -- Nautilus Data protocol ------------------------------------------------

    @property
    def ts_event(self) -> int:
        return self._ts_event

    @property
    def ts_init(self) -> int:
        return self._ts_init

    # -- Derived properties ----------------------------------------------------

    @property
    def total_score(self) -> int:
        """Combined score of both teams."""
        return self.home_score + self.away_score

    @property
    def score_diff(self) -> int:
        """Home score minus away score (positive = home leading)."""
        return self.home_score - self.away_score

    @property
    def is_close_game(self) -> bool:
        """True when the absolute score differential is 5 or fewer."""
        return abs(self.score_diff) <= 5

    @property
    def period_display(self) -> str:
        """Human-readable period + clock string, e.g. ``'Q3 5:42'``."""
        if self.is_final:
            return "FINAL"
        if self.is_halftime:
            return "HALF"
        minutes = int(self.game_clock_secs) // 60
        seconds = int(self.game_clock_secs) % 60
        return f"Q{self.period} {minutes}:{seconds:02d}"

    def __repr__(self) -> str:
        return (
            f"BasketballContextData("
            f"{self.away_team} {self.away_score} @ {self.home_team} {self.home_score}, "
            f"{self.period_display})"
        )


# ---------------------------------------------------------------------------
# Formula 1
# ---------------------------------------------------------------------------


class F1TelemetryData(Data):
    """Live telemetry snapshot for a single F1 driver."""

    def __init__(
        self,
        *,
        driver: str,
        position: int,
        lap: int,
        total_laps: int,
        gap_to_leader_secs: float,
        tire_compound: str,
        pit_count: int,
        is_pit_window: bool = False,
        ts_event: int = 0,
        ts_init: int = 0,
    ) -> None:
        self.driver = driver
        self.position = position
        self.lap = lap
        self.total_laps = total_laps
        self.gap_to_leader_secs = gap_to_leader_secs
        self.tire_compound = tire_compound
        self.pit_count = pit_count
        self.is_pit_window = is_pit_window
        self._ts_event = ts_event
        self._ts_init = ts_init

    # -- Nautilus Data protocol ------------------------------------------------

    @property
    def ts_event(self) -> int:
        return self._ts_event

    @property
    def ts_init(self) -> int:
        return self._ts_init

    # -- Derived properties ----------------------------------------------------

    @property
    def laps_remaining(self) -> int:
        """Laps left in the race (never negative)."""
        return max(0, self.total_laps - self.lap)

    @property
    def race_progress(self) -> float:
        """Fraction of race completed (0.0 .. 1.0). Returns 0.0 when total_laps is 0."""
        if self.total_laps <= 0:
            return 0.0
        return min(1.0, self.lap / self.total_laps)

    @property
    def is_late_race(self) -> bool:
        """True when more than 75% of the race is complete."""
        return self.race_progress > 0.75

    def __repr__(self) -> str:
        return (
            f"F1TelemetryData("
            f"{self.driver} P{self.position}, "
            f"lap {self.lap}/{self.total_laps}, "
            f"gap {self.gap_to_leader_secs:.3f}s, "
            f"{self.tire_compound})"
        )


# ---------------------------------------------------------------------------
# DataType helpers
# ---------------------------------------------------------------------------


def basketball_data_type(*, event_id: str = "") -> DataType:
    """Create a ``DataType`` for basketball context subscriptions."""
    metadata: dict = {"sport": "basketball"}
    if event_id:
        metadata["event_id"] = event_id
    return DataType(type=BasketballContextData, metadata=metadata)


def f1_data_type(*, session: str = "race") -> DataType:
    """Create a ``DataType`` for F1 telemetry subscriptions."""
    return DataType(type=F1TelemetryData, metadata={"sport": "f1", "session": session})
