"""NBA Fair Value strategy ported to NautilusTrader.

Port of tradingutils' nba_fair_value strategy. Follows the same
NautilusStrategy pattern as NCAABFairValueStrategy — identical alpha
(Massey/Keener 4-method Brier-weighted ensemble) adapted for NBA
game-winner markets (KXNBA-* tickers).

Alpha logic:
  1. Pre-compute 4-method rating ensemble from historical game DB
  2. On each trade tick: compute fair value, detect edge vs market price
  3. Enter when |edge| >= min_edge_cents (buy YES if undervalued)
  4. Exit via convergence: when price reaches FV - discount
  5. Kelly sizing on position
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Optional

from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from strategies.ncaab_fair_value import _RatingEnsemble, _logistic_win_prob


class NBAFairValueConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(1)
    min_edge_cents: int = 15
    convergence_discount_cents: int = 2
    kelly_fraction: float = 0.5
    max_kelly_fraction: float = 0.20
    max_entry_price: float = 0.90
    ratings_json_path: str = ""
    home_team: str = ""
    away_team: str = ""
    market_team: str = ""


class NBAFairValueStrategy(Strategy):
    """Long-only fair value strategy for NBA game winner markets."""

    def __init__(self, config: NBAFairValueConfig) -> None:
        super().__init__(config)
        self._instrument = None
        self._ensemble: Optional[_RatingEnsemble] = None
        self._pending: bool = False
        self._entry_price: float | None = None
        self._in_pos: bool = False
        self._fair_value_prob: float = 0.5
        self._tick_count: int = 0

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found")
            self.stop()
            return

        if self.config.ratings_json_path:
            path = Path(self.config.ratings_json_path)
            if path.exists():
                with open(path) as f:
                    self._ensemble = _RatingEnsemble(json.load(f))
                self.log.info(f"Loaded ratings from {path}")
            else:
                self.log.warning(f"Ratings file not found: {path}")

        if self._ensemble and self.config.home_team and self.config.away_team:
            self._fair_value_prob = self._ensemble.predict(
                self.config.home_team, self.config.away_team
            )
            self.log.info(
                f"Fair value P(home={self.config.home_team} wins) = "
                f"{self._fair_value_prob:.1%}"
            )

        self.subscribe_trade_ticks(self.config.instrument_id)

    def _is_market_team_home(self) -> bool:
        return self.config.market_team == self.config.home_team

    def _fair_yes_prob(self) -> float:
        if self._is_market_team_home():
            return self._fair_value_prob
        return 1.0 - self._fair_value_prob

    def _kelly_size(self, fair_prob: float, entry_price_cents: int) -> int:
        if entry_price_cents <= 0 or entry_price_cents >= 100:
            return 1
        market_prob = entry_price_cents / 100.0
        edge = fair_prob - market_prob
        if edge <= 0:
            return 1
        odds = (1.0 - market_prob) / market_prob
        kelly = edge * odds if odds > 0 else 0
        kelly_capped = min(kelly * self.config.kelly_fraction, self.config.max_kelly_fraction)
        return max(1, int(kelly_capped * 100))

    def on_trade_tick(self, tick: TradeTick) -> None:
        price = float(tick.price)
        self._tick_count += 1

        if self._pending or self._in_pos:
            if self._in_pos and self._entry_price is not None:
                fair_yes = self._fair_yes_prob()
                fair_cents = round(fair_yes * 100)
                convergence_target = (fair_cents - self.config.convergence_discount_cents) / 100.0
                convergence_target = max(convergence_target, self._entry_price + 0.01)
                if price >= convergence_target:
                    self._submit_exit()
            return

        if self._ensemble is None:
            return

        fair_yes = self._fair_yes_prob()
        price_cents = round(price * 100)
        fair_cents = round(fair_yes * 100)
        edge_cents = fair_cents - price_cents

        if abs(edge_cents) < self.config.min_edge_cents:
            return

        if price > self.config.max_entry_price:
            return

        if edge_cents > 0:
            qty = self._kelly_size(fair_yes, price_cents)
            quantity = self._instrument.make_qty(float(qty), round_down=True)
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=quantity,
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
            self._pending = True
            self.log.info(
                f"ENTRY: BUY {qty}x @ ~{price_cents}c, FV={fair_cents}c, edge={edge_cents}c"
            )

    def _submit_exit(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self._pending = True

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.order_side == OrderSide.BUY:
            self._entry_price = float(event.last_px)
            self._in_pos = True
        else:
            self._entry_price = None
            self._in_pos = False
        self._pending = False

    def on_order_rejected(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_canceled(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_expired(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.log.info(f"Strategy stopped after processing {self._tick_count} ticks")

    def on_reset(self) -> None:
        self._pending = False
        self._entry_price = None
        self._in_pos = False
        self._tick_count = 0
        self._instrument = None
