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
Legacy strategy shim — lets tradingutils ``I_Strategy`` signal functions
run inside the Nautilus backtest engine without a full port.

A signal function receives a :class:`MarketSnapshot` and returns an
optional :class:`SignalResult`.  Two concrete shim strategies wrap
this contract:

* :class:`LegacySignalShim` — subscribes to trade ticks.
* :class:`QuoteTickLegacyShim` — subscribes to quote ticks (real BBO).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Optional

from nautilus_trader.model.data import QuoteTick, TradeTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import StrategyConfig

from strategies.core import LongOnlyPredictionMarketStrategy


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class MarketSnapshot:
    """Input fed to a legacy signal function on every tick."""

    price: float  # last trade price (0.0-1.0 scale)
    bid: float  # best bid estimate
    ask: float  # best ask estimate
    spread: float  # ask - bid
    tick_count: int  # ticks since strategy start
    in_position: bool  # whether we currently hold a position
    entry_price: float  # avg entry price (0.0 if not in position)
    unrealized_pnl: float  # current price - entry_price (0.0 if flat)


@dataclass
class SignalResult:
    """Output from a legacy signal function."""

    action: str  # "enter", "exit", or "hold"
    strength: float = 1.0  # 0.0-1.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class LegacyShimConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(1)
    take_profit: float = 0.0  # price-distance for TP, 0 = disabled
    stop_loss: float = 0.0  # price-distance for SL, 0 = disabled


# ---------------------------------------------------------------------------
# Trade-tick shim
# ---------------------------------------------------------------------------


class LegacySignalShim(LongOnlyPredictionMarketStrategy):
    """
    Wraps a ``signal_fn(MarketSnapshot) -> Optional[SignalResult]`` so that
    legacy signal logic can run inside the Nautilus backtest engine.

    Subscribes to **trade ticks**.  For real BBO data use
    :class:`QuoteTickLegacyShim` instead.
    """

    def __init__(
        self,
        config: LegacyShimConfig,
        signal_fn: Callable[[MarketSnapshot], Optional[SignalResult]],
    ) -> None:
        super().__init__(config)
        self._signal_fn = signal_fn
        self._tick_count: int = 0

    def _subscribe(self) -> None:
        self.subscribe_trade_ticks(self.config.instrument_id)

    def _build_snapshot(self, price: float, bid: float, ask: float) -> MarketSnapshot:
        in_position = self._in_position()
        entry_price = self._entry_price if self._entry_price is not None else 0.0
        unrealized_pnl = (price - entry_price) if in_position else 0.0
        return MarketSnapshot(
            price=price,
            bid=bid,
            ask=ask,
            spread=ask - bid,
            tick_count=self._tick_count,
            in_position=in_position,
            entry_price=entry_price,
            unrealized_pnl=unrealized_pnl,
        )

    def _process_signal(self, result: Optional[SignalResult], price: float) -> None:
        if result is None:
            return

        if result.action == "enter" and not self._in_position():
            self._submit_entry(reference_price=price)
        elif result.action == "exit" and self._in_position():
            self._submit_exit()
        # "hold" or unrecognised action -> no-op

    def on_trade_tick(self, tick: TradeTick) -> None:
        price = float(tick.price)
        self._tick_count += 1

        if self._pending:
            return

        # Risk exit takes priority
        if self._risk_exit(
            price=price,
            take_profit=self.config.take_profit,
            stop_loss=self.config.stop_loss,
        ):
            return

        snapshot = self._build_snapshot(price=price, bid=price, ask=price)
        result = self._signal_fn(snapshot)
        self._process_signal(result, price)

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        super().on_order_filled(event)

    def on_reset(self) -> None:
        super().on_reset()
        self._tick_count = 0


# ---------------------------------------------------------------------------
# Quote-tick shim
# ---------------------------------------------------------------------------


class QuoteTickLegacyShim(LongOnlyPredictionMarketStrategy):
    """
    Same as :class:`LegacySignalShim` but subscribes to **quote ticks**
    so that :attr:`MarketSnapshot.bid` / :attr:`MarketSnapshot.ask` reflect
    the actual BBO rather than trade-price approximations.
    """

    def __init__(
        self,
        config: LegacyShimConfig,
        signal_fn: Callable[[MarketSnapshot], Optional[SignalResult]],
    ) -> None:
        super().__init__(config)
        self._signal_fn = signal_fn
        self._tick_count: int = 0

    def _subscribe(self) -> None:
        self.subscribe_quote_ticks(self.config.instrument_id)

    def _build_snapshot(self, price: float, bid: float, ask: float) -> MarketSnapshot:
        in_position = self._in_position()
        entry_price = self._entry_price if self._entry_price is not None else 0.0
        unrealized_pnl = (price - entry_price) if in_position else 0.0
        return MarketSnapshot(
            price=price,
            bid=bid,
            ask=ask,
            spread=ask - bid,
            tick_count=self._tick_count,
            in_position=in_position,
            entry_price=entry_price,
            unrealized_pnl=unrealized_pnl,
        )

    def _process_signal(self, result: Optional[SignalResult], price: float) -> None:
        if result is None:
            return

        if result.action == "enter" and not self._in_position():
            self._submit_entry(reference_price=price, visible_size=None)
        elif result.action == "exit" and self._in_position():
            self._submit_exit()

    def on_quote_tick(self, tick: QuoteTick) -> None:
        bid = float(tick.bid_price)
        ask = float(tick.ask_price)
        price = (bid + ask) / 2.0
        self._tick_count += 1

        if self._pending:
            return

        # Risk exit takes priority
        if self._risk_exit(
            price=price,
            take_profit=self.config.take_profit,
            stop_loss=self.config.stop_loss,
        ):
            return

        snapshot = self._build_snapshot(price=price, bid=bid, ask=ask)
        result = self._signal_fn(snapshot)
        self._process_signal(result, price)

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        super().on_order_filled(event)

    def on_reset(self) -> None:
        super().on_reset()
        self._tick_count = 0
