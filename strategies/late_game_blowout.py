"""Late Game Blowout strategy ported to NautilusTrader.

Port of tradingutils' late_game_blowout_strategy. Without live score feeds,
this models blowout detection via extreme market prices: a contract trading
above blowout_price_threshold (default 85c) implies a commanding lead late
in the game.

Alpha logic:
  1. On each trade tick, check if price crosses into the blowout zone
  2. Entry when blowout_price_threshold <= price <= max_entry_price
  3. Fair value assumed at historical blowout win rate (default 95c)
  4. Edge = fair_value_cents - price_cents; enter when >= min_edge_cents
  5. Kelly sizing based on edge to settlement at 100c
  6. Hold to settlement — no convergence exit
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from nautilus_trader.model.data import QuoteTick, TradeTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import StrategyConfig

from strategies.core import LongOnlyPredictionMarketStrategy


class TradeTickLateGameBlowoutConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(1)
    blowout_price_threshold: float = 0.85
    max_entry_price: float = 0.95
    min_edge_cents: int = 3
    kelly_fraction: float = 0.5
    max_kelly_fraction: float = 0.20
    max_positions: int = 3
    blowout_win_rate: float = 0.95


class QuoteTickLateGameBlowoutConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(1)
    blowout_price_threshold: float = 0.85
    max_entry_price: float = 0.95
    min_edge_cents: int = 3
    kelly_fraction: float = 0.5
    max_kelly_fraction: float = 0.20
    max_positions: int = 3
    blowout_win_rate: float = 0.95


def _blowout_kelly_size(
    fair_prob: float,
    entry_price_cents: int,
    kelly_fraction: float,
    max_kelly_fraction: float,
) -> int:
    """Compute Kelly-sized quantity for a blowout entry.

    Returns an integer contract count (minimum 1).
    """
    if entry_price_cents <= 0 or entry_price_cents >= 100:
        return 1
    market_prob = entry_price_cents / 100.0
    edge = fair_prob - market_prob
    if edge <= 0:
        return 1
    odds = (1.0 - market_prob) / market_prob
    kelly = edge * odds if odds > 0 else 0
    kelly_capped = min(kelly * kelly_fraction, max_kelly_fraction)
    return max(1, int(kelly_capped * 100))


class _LateGameBlowoutBase(LongOnlyPredictionMarketStrategy):
    """Buy contracts in the blowout price zone and hold to settlement.

    A price above *blowout_price_threshold* signals a commanding in-game
    lead. We enter when the market-implied probability is high but still
    offers edge relative to the historical blowout win rate. The position
    is held until settlement — no exit logic.
    """

    def __init__(
        self,
        config: TradeTickLateGameBlowoutConfig | QuoteTickLateGameBlowoutConfig,
    ) -> None:
        super().__init__(config)
        self._position_count: int = 0
        self._tick_count: int = 0

    def _on_price(
        self,
        price: float,
        *,
        entry_price: Optional[float] = None,
        visible_size: Optional[float] = None,
    ) -> None:
        self._tick_count += 1

        if self._pending:
            return

        # Already at max positions — do nothing
        if self._position_count >= int(self.config.max_positions):
            return

        # Price must be in the blowout zone
        if price < float(self.config.blowout_price_threshold):
            return
        if price > float(self.config.max_entry_price):
            return

        # Edge calculation
        price_cents = round(price * 100)
        fair_value_cents = round(float(self.config.blowout_win_rate) * 100)
        edge_cents = fair_value_cents - price_cents

        if edge_cents < int(self.config.min_edge_cents):
            return

        self.log.info(
            f"BLOWOUT ENTRY: price={price_cents}c, "
            f"fair={fair_value_cents}c, edge={edge_cents}c, "
            f"positions={self._position_count}/{self.config.max_positions}"
        )

        self._submit_entry(
            reference_price=price if entry_price is None else entry_price,
            visible_size=visible_size,
        )

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        super().on_order_filled(event)
        if event.order_side == OrderSide.BUY:
            self._position_count += 1

    def on_stop(self) -> None:
        # Leave filled positions open so the runner marks them to settlement.
        self.cancel_all_orders(self.config.instrument_id)
        self.log.info(
            f"Late Game Blowout stopped after {self._tick_count} ticks, "
            f"{self._position_count} entries"
        )

    def on_reset(self) -> None:
        super().on_reset()
        self._position_count = 0
        self._tick_count = 0


class TradeTickLateGameBlowoutStrategy(_LateGameBlowoutBase):
    def _subscribe(self) -> None:
        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_trade_tick(self, tick: TradeTick) -> None:
        price = float(tick.price)
        self._on_price(price, entry_price=price)


class QuoteTickLateGameBlowoutStrategy(_LateGameBlowoutBase):
    def _subscribe(self) -> None:
        self.subscribe_quote_ticks(self.config.instrument_id)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        mid = (float(tick.bid_price) + float(tick.ask_price)) / 2.0
        self._on_price(
            mid,
            entry_price=float(tick.ask_price),
            visible_size=float(tick.ask_size),
        )
