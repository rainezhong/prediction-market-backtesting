"""NBA Q4 Total Points strategy ported to NautilusTrader.

Simplified port of tradingutils' nba_q4_totals_strategy (reactive mode).
Without live score feeds, models the alpha as a price-momentum post-spike
entry on near-strike total-points contracts.

Original alpha (reactive mode):
  - Detect FG scoring events via total score jumps
  - Exploit stale Kalshi orderbooks after scoring events
  - Buy YES on near-strike contracts at stale mid price
  - Q4 only, FG-only, YES-only, 3-7 min remaining, 40-70c entry, spread <= 3c
  - Hold to settlement: 86% WR, +31c/trade average

Simplified (Nautilus) approach:
  1. Track rolling window of recent trade prices
  2. Detect upward price spike (proxy for scoring event pushing total
     closer to over): current_price - price_N_ticks_ago >= spike_threshold
  3. If spike detected AND price in sweet spot [40c, 70c]: BUY YES
  4. Estimate fair value: current_price + momentum continuation (+5c)
  5. Edge = fair_estimate - current_price; gate on min_edge_cents
  6. Kelly sizing on entry
  7. YES-only (NO side is structurally negative: -11.5c/trade)
  8. Hold to settlement
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Optional

from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy, StrategyConfig


class NBAQ4TotalsConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    """Configuration for the NBA Q4 Total Points strategy."""

    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(1)
    spike_threshold_cents: int = 3          # price spike to detect scoring event
    spike_window_ticks: int = 5             # window to detect spike
    min_entry_price: float = 0.40           # sweet spot lower bound
    max_entry_price: float = 0.70           # sweet spot upper bound
    momentum_continuation_cents: int = 5    # expected post-spike drift
    min_edge_cents: int = 3                 # minimum edge to enter
    kelly_fraction: float = 0.5
    max_kelly_fraction: float = 0.20
    max_positions: int = 5                  # max concurrent positions


class NBAQ4TotalsStrategy(Strategy):
    """YES-only price-momentum strategy for NBA Q4 total-points markets.

    Detects upward price spikes (proxy for scoring events that push the
    total closer to the over line) and buys YES contracts in the 40-70c
    sweet spot where stale orderbooks offer mispricing.  Holds to
    settlement -- Q4 games end soon after entry.

    YES-only filter is safety-critical: the NO side is structurally
    unprofitable (-11.5c/trade in backtest).
    """

    def __init__(self, config: NBAQ4TotalsConfig) -> None:
        super().__init__(config)
        self._instrument = None

        # Rolling price window for spike detection
        self._price_window: deque[float] = deque(maxlen=config.spike_window_ticks)

        # Position tracking
        self._pending: bool = False
        self._in_pos: bool = False
        self._entry_price: Optional[float] = None
        self._position_count: int = 0

        # Diagnostics
        self._tick_count: int = 0
        self._spike_count: int = 0
        self._entry_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found")
            self.stop()
            return

        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.log.info(
            f"Strategy stopped: {self._tick_count} ticks, "
            f"{self._spike_count} spikes detected, "
            f"{self._entry_count} entries"
        )

    def on_reset(self) -> None:
        self._instrument = None
        self._price_window.clear()
        self._pending = False
        self._in_pos = False
        self._entry_price = None
        self._position_count = 0
        self._tick_count = 0
        self._spike_count = 0
        self._entry_count = 0

    # ------------------------------------------------------------------
    # Spike detection
    # ------------------------------------------------------------------

    def _detect_spike(self, current_price_cents: int) -> bool:
        """Return True if an upward spike >= threshold is detected.

        Compares the current price to the oldest price in the rolling
        window.  Only detects *upward* spikes (YES-only).
        """
        if len(self._price_window) < self.config.spike_window_ticks:
            return False

        oldest_price_cents = round(self._price_window[0] * 100)
        return (current_price_cents - oldest_price_cents) >= self.config.spike_threshold_cents

    # ------------------------------------------------------------------
    # Kelly sizing
    # ------------------------------------------------------------------

    def _kelly_size(self, fair_prob: float, entry_price_cents: int) -> int:
        """Compute Kelly-sized position count.

        Args:
            fair_prob: Estimated probability of YES settling (0-1).
            entry_price_cents: Entry price in cents (1-99).

        Returns:
            Integer contract count, minimum 1.
        """
        if entry_price_cents <= 0 or entry_price_cents >= 100:
            return 1

        market_prob = entry_price_cents / 100.0
        edge = fair_prob - market_prob
        if edge <= 0:
            return 1

        odds = (1.0 - market_prob) / market_prob
        kelly = edge * odds if odds > 0 else 0
        kelly_capped = min(
            kelly * self.config.kelly_fraction,
            self.config.max_kelly_fraction,
        )
        return max(1, int(kelly_capped * 100))

    # ------------------------------------------------------------------
    # Core tick handler
    # ------------------------------------------------------------------

    def on_trade_tick(self, tick: TradeTick) -> None:
        price = float(tick.price)
        price_cents = round(price * 100)
        self._tick_count += 1

        # Detect spike *before* adding current price to window
        spike_detected = self._detect_spike(price_cents)

        # Update rolling price window
        self._price_window.append(price)

        if spike_detected:
            self._spike_count += 1

        # Skip if pending order or at max positions
        if self._pending:
            return

        if self._position_count >= self.config.max_positions:
            return

        # Only enter on detected spike
        if not spike_detected:
            return

        # Sweet-spot price filter: 40-70c (YES-only)
        if price < self.config.min_entry_price or price > self.config.max_entry_price:
            return

        # Estimate fair value: current price + momentum continuation
        fair_cents = price_cents + self.config.momentum_continuation_cents
        fair_cents = min(fair_cents, 99)  # cap at 99c
        fair_prob = fair_cents / 100.0

        # Edge gate
        edge_cents = fair_cents - price_cents
        if edge_cents < self.config.min_edge_cents:
            return

        # Kelly sizing
        qty = self._kelly_size(fair_prob, price_cents)
        quantity = self._instrument.make_qty(float(qty), round_down=True)

        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=quantity,
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order)
        self._pending = True
        self._entry_count += 1

        self.log.info(
            f"ENTRY BUY: {qty}x @ ~{price_cents}c, "
            f"FV={fair_cents}c, edge={edge_cents}c, "
            f"positions={self._position_count + 1}/{self.config.max_positions}"
        )

    # ------------------------------------------------------------------
    # Order event handlers
    # ------------------------------------------------------------------

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.order_side == OrderSide.BUY:
            self._entry_price = float(event.last_px)
            self._in_pos = True
            self._position_count += 1
        else:
            # Exit fill (settlement or close)
            self._position_count = max(0, self._position_count - 1)
            if self._position_count == 0:
                self._entry_price = None
                self._in_pos = False
        self._pending = False

    def on_order_rejected(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_canceled(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_expired(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False
