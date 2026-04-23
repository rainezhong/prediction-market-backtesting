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
Demo strategy combining trade ticks with basketball context data.

Subscribes to both ``TradeTick`` and ``BasketballContextData`` and enters
when the losing team's contract is cheap in the late game.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from nautilus_trader.model.data import CustomData
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import StrategyConfig

from prediction_market_extensions.data.custom_types import (
    BasketballContextData,
    basketball_data_type,
)
from strategies.core import LongOnlyPredictionMarketStrategy


class BasketballContextDemoConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(1)
    event_id: str = ""


class BasketballContextDemo(LongOnlyPredictionMarketStrategy):
    """Demo: enters when score differential exceeds threshold during trade tick."""

    def __init__(self, config: BasketballContextDemoConfig) -> None:
        super().__init__(config)
        self._latest_context: Optional[BasketballContextData] = None

    def _subscribe(self) -> None:
        self.subscribe_trade_ticks(self.config.instrument_id)
        self.subscribe_data(basketball_data_type(event_id=self.config.event_id))

    def on_data(self, data: object) -> None:
        if isinstance(data, CustomData) and isinstance(data.data, BasketballContextData):
            self._latest_context = data.data

    def on_trade_tick(self, tick: object) -> None:
        if self._pending or self._latest_context is None:
            return
        ctx = self._latest_context
        price = float(tick.price)
        # Example: buy when losing team's price drops below threshold in late game
        if ctx.period >= 4 and ctx.score_diff > 10 and price < 0.15 and not self._in_position():
            self._submit_entry(reference_price=price)
