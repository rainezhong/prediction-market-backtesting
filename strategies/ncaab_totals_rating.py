"""NCAAB Totals Rating strategy ported to NautilusTrader.

Port of tradingutils' ncaab_totals_rating strategy. Uses Massey pace
ratings to predict game totals, then enters when the predicted total
diverges significantly from the market strike.

Alpha logic:
  1. Load pre-computed pace ratings + league average from JSON
  2. On each trade tick: predicted_total = league_avg + pace_away + pace_home
  3. If |predicted - strike| >= min_edge_points: enter position
  4. P(over) = 1 - Phi((strike - predicted) / sigma) using normal CDF
  5. BUY if over, SELL if under (on the single instrument)
  6. Half-Kelly sizing
  7. Hold to settlement
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Optional

from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy, StrategyConfig


# ---------------------------------------------------------------------------
# Normal CDF helper (Abramowitz & Stegun approximation)
# ---------------------------------------------------------------------------

def _normal_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun polynomial approx."""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(
        -x * x / 2
    )
    return 0.5 * (1.0 + sign * y)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class NCAABTotalsRatingConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(1)
    min_edge_points: float = 5.0
    sigma: float = 15.0
    kelly_fraction: float = 0.5
    max_kelly_fraction: float = 0.20
    max_entry_price: float = 0.90
    ratings_json_path: str = ""
    strike: float = 0.0
    away_team: str = ""
    home_team: str = ""


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class NCAABTotalsRatingStrategy(Strategy):
    """Pregame NCAAB totals strategy using Massey pace ratings.

    Predicts game totals from pre-computed pace data, compares to market
    strike, and enters when the predicted total diverges significantly.
    Holds to settlement (no convergence exit).
    """

    def __init__(self, config: NCAABTotalsRatingConfig) -> None:
        super().__init__(config)
        self._instrument = None
        self._pace_ratings: dict[str, float] = {}
        self._league_avg: float = 0.0
        self._predicted_total: float | None = None
        self._pending: bool = False
        self._in_pos: bool = False
        self._tick_count: int = 0

    # ----- lifecycle -----

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found")
            self.stop()
            return

        # Load pre-computed pace ratings from JSON
        if self.config.ratings_json_path:
            path = Path(self.config.ratings_json_path)
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                self._pace_ratings = data.get("pace_ratings", {})
                self._league_avg = data.get("league_avg", 0.0)
                self.log.info(
                    f"Loaded pace ratings: {len(self._pace_ratings)} teams, "
                    f"league_avg={self._league_avg:.1f}"
                )
            else:
                self.log.warning(f"Ratings file not found: {path}")

        # Pre-compute predicted total
        if self._pace_ratings and self.config.away_team and self.config.home_team:
            pace_away = self._pace_ratings.get(self.config.away_team, 0.0)
            pace_home = self._pace_ratings.get(self.config.home_team, 0.0)
            self._predicted_total = self._league_avg + pace_away + pace_home
            self.log.info(
                f"Predicted total: {self._predicted_total:.1f} "
                f"(strike={self.config.strike}, "
                f"{self.config.away_team} @ {self.config.home_team})"
            )

        self.subscribe_trade_ticks(self.config.instrument_id)

    # ----- alpha helpers -----

    def _over_probability(self) -> float | None:
        """P(over) = 1 - Phi((strike - predicted) / sigma)."""
        if self._predicted_total is None:
            return None
        sigma = self.config.sigma
        if sigma <= 0:
            return 1.0 if self._predicted_total > self.config.strike else 0.0
        z = (self.config.strike - self._predicted_total) / sigma
        return 1.0 - _normal_cdf(z)

    def _kelly_size(self, fair_prob: float, entry_price_cents: int) -> int:
        """Compute half-Kelly position size in contracts."""
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

    # ----- tick handling -----

    def on_trade_tick(self, tick: TradeTick) -> None:
        price = float(tick.price)
        self._tick_count += 1

        # Already in position or order pending -- hold to settlement
        if self._pending or self._in_pos:
            return

        if self._predicted_total is None:
            return

        edge = self._predicted_total - self.config.strike

        if abs(edge) < self.config.min_edge_points:
            return

        if price > self.config.max_entry_price:
            return

        p_over = self._over_probability()
        if p_over is None:
            return

        price_cents = round(price * 100)

        if edge > 0:
            # Predicted above strike -> over -> BUY
            side = OrderSide.BUY
            fair_prob = p_over
            qty = self._kelly_size(fair_prob, price_cents)
            label = "OVER"
        else:
            # Predicted below strike -> under -> SELL
            side = OrderSide.SELL
            fair_prob = 1.0 - p_over
            no_price_cents = 100 - price_cents
            qty = self._kelly_size(fair_prob, no_price_cents)
            label = "UNDER"

        quantity = self._instrument.make_qty(float(qty), round_down=True)
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=quantity,
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order)
        self._pending = True
        self.log.info(
            f"ENTRY: {label} {side.name} {qty}x @ ~{price_cents}c, "
            f"predicted={self._predicted_total:.1f}, strike={self.config.strike}, "
            f"edge={edge:+.1f}pts, P({label.lower()})={fair_prob:.1%}"
        )

    # ----- order events -----

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        self._in_pos = True
        self._pending = False

    def on_order_rejected(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_canceled(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_expired(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    # ----- shutdown -----

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.log.info(f"Strategy stopped after processing {self._tick_count} ticks")

    def on_reset(self) -> None:
        self._pending = False
        self._in_pos = False
        self._tick_count = 0
        self._instrument = None
        self._predicted_total = None
