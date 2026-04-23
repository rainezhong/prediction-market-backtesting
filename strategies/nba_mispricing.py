"""NBA Mispricing strategy ported to NautilusTrader.

Port of tradingutils' nba_mispricing_strategy. The original relies on
live ESPN score feeds to compute score-implied fair value. Since
Nautilus backtests operate on trade-tick data only, this port uses a
**price-impulse mean-reversion** proxy that captures the same edge:

NBA game markets move sharply on scoring events. When price overshoots,
it tends to mean-revert. This strategy detects impulse moves in a
rolling tick window and enters in the reversion direction.

Alpha logic:
  1. Maintain a rolling window of the last N trade prices
  2. Detect impulse: price change >= threshold over the window
  3. Enter opposite to the impulse (mean reversion)
  4. Kelly sizing based on impulse magnitude
  5. Exit when price reverts toward pre-impulse level (convergence)
  6. Cooldown between entries to avoid overtrading
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Optional

from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy, StrategyConfig


class NBAMispricingConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(1)
    min_edge_cents: int = 3
    impulse_threshold_cents: int = 5
    impulse_window_ticks: int = 10
    convergence_discount_cents: int = 2
    max_entry_price: float = 0.90
    kelly_fraction: float = 0.25
    max_kelly_fraction: float = 0.25
    cooldown_ticks: int = 30
    max_position_per_game: int = 5


class NBAMispricingStrategy(Strategy):
    """Mean-reversion strategy for NBA game markets based on price impulses.

    Detects sharp price moves (impulses) caused by scoring events and
    trades the expected reversion back toward the pre-impulse level.
    """

    def __init__(self, config: NBAMispricingConfig) -> None:
        super().__init__(config)
        self._instrument = None
        self._pending: bool = False
        self._entry_price: Optional[float] = None
        self._entry_side: Optional[OrderSide] = None
        self._in_pos: bool = False
        self._tick_count: int = 0
        self._ticks_since_entry: int = 0
        self._last_entry_tick: int = 0
        self._entry_count: int = 0
        self._pre_impulse_price: Optional[float] = None

        # Rolling price window
        self._price_window: deque[float] = deque(
            maxlen=config.impulse_window_ticks
        )

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(
                f"Instrument {self.config.instrument_id} not found"
            )
            self.stop()
            return

        self.subscribe_trade_ticks(self.config.instrument_id)

    def _detect_impulse(self) -> Optional[tuple[float, str]]:
        """Detect a price impulse in the rolling window.

        Returns:
            Tuple of (impulse_cents, direction) where direction is
            "dump" (price fell) or "pump" (price rose), or None if
            no impulse detected.
        """
        if len(self._price_window) < self._price_window.maxlen:
            return None

        first_price = self._price_window[0]
        last_price = self._price_window[-1]
        delta_cents = round((last_price - first_price) * 100)

        if delta_cents <= -self.config.impulse_threshold_cents:
            # Price dumped -- expect mean reversion up
            return (abs(delta_cents), "dump")
        elif delta_cents >= self.config.impulse_threshold_cents:
            # Price pumped -- expect mean reversion down
            return (abs(delta_cents), "pump")
        return None

    def _kelly_size(self, impulse_cents: int) -> int:
        """Compute Kelly-based position size from impulse magnitude.

        Larger impulses suggest larger mispricings and therefore
        higher edge, warranting bigger positions.
        """
        if impulse_cents <= 0:
            return 1
        # Edge proxy: impulse magnitude as fraction of 100c
        edge = impulse_cents / 100.0
        # Conservative odds assumption: even money
        kelly = edge * 1.0
        kelly_capped = min(
            kelly * self.config.kelly_fraction,
            self.config.max_kelly_fraction,
        )
        return max(1, int(kelly_capped * 100))

    def on_trade_tick(self, tick: TradeTick) -> None:
        price = float(tick.price)
        self._tick_count += 1
        self._price_window.append(price)

        # --- Exit logic ---
        if self._pending:
            return

        if self._in_pos and self._entry_price is not None:
            self._ticks_since_entry += 1
            if self._should_exit(price):
                self._submit_exit()
            return

        # --- Entry logic ---

        # Cooldown check
        if (self._tick_count - self._last_entry_tick) < self.config.cooldown_ticks:
            return

        # Max positions check
        if self._entry_count >= self.config.max_position_per_game:
            return

        # Price bounds check
        if price > self.config.max_entry_price or price < (1.0 - self.config.max_entry_price):
            return

        impulse = self._detect_impulse()
        if impulse is None:
            return

        impulse_cents, direction = impulse

        # Edge must exceed minimum
        if impulse_cents < self.config.min_edge_cents:
            return

        # Record pre-impulse price for convergence exit
        self._pre_impulse_price = self._price_window[0]

        qty = self._kelly_size(impulse_cents)
        quantity = self._instrument.make_qty(float(qty), round_down=True)

        if direction == "dump":
            # Price dumped -- buy YES expecting reversion up
            side = OrderSide.BUY
            self.log.info(
                f"IMPULSE ENTRY: BUY {qty}x @ ~{round(price * 100)}c, "
                f"dump={impulse_cents}c, revert target={round(self._pre_impulse_price * 100)}c"
            )
        else:
            # Price pumped -- sell (buy NO side) expecting reversion down
            side = OrderSide.SELL
            self.log.info(
                f"IMPULSE ENTRY: SELL {qty}x @ ~{round(price * 100)}c, "
                f"pump={impulse_cents}c, revert target={round(self._pre_impulse_price * 100)}c"
            )

        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=quantity,
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order)
        self._pending = True

    def _should_exit(self, price: float) -> bool:
        """Check convergence exit condition.

        Exit when price reverts back toward the pre-impulse level,
        minus a small discount to lock in profit.
        """
        if self._pre_impulse_price is None or self._entry_price is None:
            return False

        discount = self.config.convergence_discount_cents / 100.0

        if self._entry_side == OrderSide.BUY:
            # Long position -- exit when price reverts up toward target
            target = self._pre_impulse_price - discount
            target = max(target, self._entry_price + 0.01)
            return price >= target
        else:
            # Short position -- exit when price reverts down toward target
            target = self._pre_impulse_price + discount
            target = min(target, self._entry_price - 0.01)
            return price <= target

    def _submit_exit(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self._pending = True

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        fill_price = float(event.last_px)
        if not self._in_pos:
            # Entry fill
            self._entry_price = fill_price
            self._entry_side = event.order_side
            self._in_pos = True
            self._ticks_since_entry = 0
            self._last_entry_tick = self._tick_count
            self._entry_count += 1
        else:
            # Exit fill
            self._entry_price = None
            self._entry_side = None
            self._in_pos = False
            self._pre_impulse_price = None
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
        self.log.info(
            f"Strategy stopped after {self._tick_count} ticks, "
            f"{self._entry_count} entries"
        )

    def on_reset(self) -> None:
        self._pending = False
        self._entry_price = None
        self._entry_side = None
        self._in_pos = False
        self._tick_count = 0
        self._ticks_since_entry = 0
        self._last_entry_tick = 0
        self._entry_count = 0
        self._pre_impulse_price = None
        self._price_window.clear()
        self._instrument = None
