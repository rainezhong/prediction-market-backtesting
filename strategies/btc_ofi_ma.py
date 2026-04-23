"""BTC OFI + Moving Average regime-filtered strategy ported to NautilusTrader.

Port of tradingutils' btc_ofi_ma_strategy. Adds a dual-EMA regime filter
on top of the OFI z-score signal used by btc_ofi_15m.

Alpha logic:
  1. Track rolling buy/sell imbalance from trade ticks (OFI proxy)
  2. Compute dual-EMA (fast/slow) on trade prices for regime detection
  3. MA regime: bullish = fast_ema > slow_ema, bearish = fast_ema < slow_ema
  4. Entry ONLY when OFI direction agrees with MA regime:
     - BUY YES: OFI bullish AND fast > slow (bullish regime) AND price < 0.50
     - SELL (BUY NO): OFI bearish AND fast < slow (bearish regime) AND price > 0.50
  5. Safety: |ofi_zscore| capped at max_abs_z_score (reject overreaction)
  6. Half-Kelly sizing, cooldown between entries
  7. Hold to settlement (15-minute windows)

Research basis (from tradingutils validation):
  - BSS 0.081 (STRONG gate), IC 0.338, WR 70.3%
  - Default EMA periods: fast=4, slow=12
  - MA regime filter removes signals that contradict the trend
"""

from __future__ import annotations

import math
from collections import deque
from decimal import Decimal
from typing import Optional

from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy, StrategyConfig


class BTCOfiMaConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    """Configuration for the BTC OFI + MA regime-filtered strategy."""

    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(1)
    ofi_window_ticks: int = 50
    ofi_threshold: float = 0.6
    ema_fast_period: int = 4
    ema_slow_period: int = 12
    max_abs_z_score: float = 2.5
    min_edge_cents: int = 3
    max_entry_price: float = 0.90
    kelly_fraction: float = 0.25
    max_kelly_fraction: float = 0.25
    max_consecutive_losses: int = 3
    cooldown_ticks: int = 20


class BTCOfiMaStrategy(Strategy):
    """BTC OFI + MA regime-filtered strategy using trade-tick OFI proxy.

    Extends the OFI-only approach (btc_ofi_15m) with a dual-EMA regime
    filter. Entries only fire when the OFI direction agrees with the
    moving average trend regime (fast EMA vs slow EMA).

    Consecutive-loss halting is safety-critical -- strategy stops emitting
    signals after ``max_consecutive_losses`` losses in a row.
    """

    def __init__(self, config: BTCOfiMaConfig) -> None:
        super().__init__(config)
        self._instrument = None

        # OFI proxy state -- rolling window of +1 (buy) / -1 (sell)
        self._ofi_window: deque[int] = deque(maxlen=config.ofi_window_ticks)
        self._ofi_value: float = 0.0

        # OFI z-score state -- rolling mean/variance for normalization
        self._ofi_history: deque[float] = deque(maxlen=config.ofi_window_ticks)
        self._ofi_zscore: float = 0.0

        # Dual-EMA state for MA regime filter
        self._ema_fast_alpha: float = 2.0 / (config.ema_fast_period + 1)
        self._ema_slow_alpha: float = 2.0 / (config.ema_slow_period + 1)
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None

        # Position tracking
        self._pending: bool = False
        self._in_pos: bool = False
        self._entry_price: Optional[float] = None
        self._entry_side: Optional[OrderSide] = None

        # Safety state
        self._consecutive_losses: int = 0
        self._halted: bool = False
        self._cooldown_remaining: int = 0
        self._tick_count: int = 0

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
            f"Strategy stopped after {self._tick_count} ticks, "
            f"consecutive_losses={self._consecutive_losses}, halted={self._halted}"
        )

    def on_reset(self) -> None:
        self._instrument = None
        self._ofi_window.clear()
        self._ofi_value = 0.0
        self._ofi_history.clear()
        self._ofi_zscore = 0.0
        self._ema_fast = None
        self._ema_slow = None
        self._pending = False
        self._in_pos = False
        self._entry_price = None
        self._entry_side = None
        self._consecutive_losses = 0
        self._halted = False
        self._cooldown_remaining = 0
        self._tick_count = 0

    # ------------------------------------------------------------------
    # OFI proxy + z-score
    # ------------------------------------------------------------------

    def _update_ofi(self, tick: TradeTick) -> None:
        """Update rolling OFI from trade tick aggressor side and compute z-score."""
        if tick.aggressor_side == AggressorSide.BUYER:
            self._ofi_window.append(1)
        elif tick.aggressor_side == AggressorSide.SELLER:
            self._ofi_window.append(-1)
        else:
            # No aggressor info -- treat as neutral
            self._ofi_window.append(0)

        if len(self._ofi_window) > 0:
            self._ofi_value = sum(self._ofi_window) / len(self._ofi_window)
        else:
            self._ofi_value = 0.0

        # Track OFI history for z-score computation
        self._ofi_history.append(self._ofi_value)

        if len(self._ofi_history) >= 2:
            n = len(self._ofi_history)
            mean = sum(self._ofi_history) / n
            variance = sum((x - mean) ** 2 for x in self._ofi_history) / n
            std = math.sqrt(variance) if variance > 0 else 1e-9
            self._ofi_zscore = (self._ofi_value - mean) / std
        else:
            self._ofi_zscore = 0.0

    # ------------------------------------------------------------------
    # EMA tracker
    # ------------------------------------------------------------------

    def _update_ema(self, price: float) -> None:
        """Update fast and slow EMAs from trade price."""
        if self._ema_fast is None:
            self._ema_fast = price
            self._ema_slow = price
        else:
            self._ema_fast = self._ema_fast_alpha * price + (1 - self._ema_fast_alpha) * self._ema_fast
            self._ema_slow = self._ema_slow_alpha * price + (1 - self._ema_slow_alpha) * self._ema_slow

    @property
    def _regime_bullish(self) -> bool:
        """True when fast EMA > slow EMA (bullish regime)."""
        if self._ema_fast is None or self._ema_slow is None:
            return False
        return self._ema_fast > self._ema_slow

    # ------------------------------------------------------------------
    # Kelly sizing
    # ------------------------------------------------------------------

    def _kelly_size(self, edge_prob: float, entry_price_cents: int) -> int:
        """Compute Kelly-sized position.

        Args:
            edge_prob: Estimated probability of winning (0-1).
            entry_price_cents: Entry price in cents (1-99).

        Returns:
            Integer contract count, minimum 1.
        """
        if entry_price_cents <= 0 or entry_price_cents >= 100:
            return 1

        market_prob = entry_price_cents / 100.0
        edge = edge_prob - market_prob
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
        self._tick_count += 1

        # Update OFI proxy + z-score
        self._update_ofi(tick)

        # Update dual-EMA regime filter
        self._update_ema(price)

        # Decrement cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # If we have a pending order or are in position, just monitor
        if self._pending or self._in_pos:
            return

        # Safety: halted after consecutive losses
        if self._halted:
            return

        # Cooldown active
        if self._cooldown_remaining > 0:
            return

        # Need enough data for OFI calculation
        if len(self._ofi_window) < self.config.ofi_window_ticks:
            return

        # Need EMA initialized
        if self._ema_fast is None or self._ema_slow is None:
            return

        # Price guard
        if price > self.config.max_entry_price:
            return

        # Safety: reject extreme z-scores (overreaction / mean-reversion)
        if abs(self._ofi_zscore) > self.config.max_abs_z_score:
            return

        ofi = self._ofi_value
        price_cents = round(price * 100)

        # BUY YES: OFI bullish AND MA regime bullish AND price undervalued
        if (
            ofi > self.config.ofi_threshold
            and self._regime_bullish
            and price < 0.50
        ):
            # Fair value estimate: use OFI magnitude as confidence proxy
            fair_prob = 0.50 + (ofi - self.config.ofi_threshold) * 0.5
            fair_prob = min(fair_prob, 0.95)
            fair_cents = round(fair_prob * 100)
            edge_cents = fair_cents - price_cents

            if edge_cents >= self.config.min_edge_cents:
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
                self.log.info(
                    f"ENTRY BUY: {qty}x @ ~{price_cents}c, "
                    f"OFI={ofi:.3f}, z={self._ofi_zscore:.2f}, "
                    f"FV={fair_cents}c, edge={edge_cents}c, "
                    f"EMA_f={self._ema_fast:.4f}, EMA_s={self._ema_slow:.4f}"
                )

        # BUY NO (SELL): OFI bearish AND MA regime bearish AND price overvalued
        elif (
            ofi < -self.config.ofi_threshold
            and not self._regime_bullish
            and price > 0.50
        ):
            fair_prob_yes = 0.50 - (abs(ofi) - self.config.ofi_threshold) * 0.5
            fair_prob_yes = max(fair_prob_yes, 0.05)
            fair_cents_yes = round(fair_prob_yes * 100)
            edge_cents = price_cents - fair_cents_yes

            if edge_cents >= self.config.min_edge_cents:
                fair_prob_no = 1.0 - fair_prob_yes
                no_price_cents = 100 - price_cents
                qty = self._kelly_size(fair_prob_no, no_price_cents)
                quantity = self._instrument.make_qty(float(qty), round_down=True)
                order = self.order_factory.market(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=quantity,
                    time_in_force=TimeInForce.IOC,
                )
                self.submit_order(order)
                self._pending = True
                self.log.info(
                    f"ENTRY SELL: {qty}x @ ~{price_cents}c, "
                    f"OFI={ofi:.3f}, z={self._ofi_zscore:.2f}, "
                    f"FV_yes={fair_cents_yes}c, edge={edge_cents}c, "
                    f"EMA_f={self._ema_fast:.4f}, EMA_s={self._ema_slow:.4f}"
                )

    # ------------------------------------------------------------------
    # Order event handlers
    # ------------------------------------------------------------------

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        if not self._in_pos:
            # Entry fill
            self._entry_price = float(event.last_px)
            self._entry_side = event.order_side
            self._in_pos = True
            self._pending = False
        else:
            # Exit fill -- evaluate win/loss for consecutive-loss tracking
            exit_price = float(event.last_px)
            if self._entry_price is not None:
                if self._entry_side == OrderSide.BUY:
                    pnl = exit_price - self._entry_price
                else:
                    pnl = self._entry_price - exit_price

                if pnl < 0:
                    self._consecutive_losses += 1
                    if self._consecutive_losses >= self.config.max_consecutive_losses:
                        self._halted = True
                        self.log.warning(
                            f"HALTED: {self._consecutive_losses} consecutive losses"
                        )
                else:
                    self._consecutive_losses = 0

            self._entry_price = None
            self._entry_side = None
            self._in_pos = False
            self._pending = False
            self._cooldown_remaining = self.config.cooldown_ticks

    def on_order_rejected(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_canceled(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_expired(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False
