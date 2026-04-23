"""BTC OFI 15-minute settlement strategy ported to NautilusTrader.

Simplified port of tradingutils' btc_ofi strategy. Uses a trade-tick-derived
OFI proxy (buy/sell imbalance over a rolling window) instead of the full
CEX L2 + UKF pipeline. Full CEX integration is a follow-up (N3.6 custom
data types).

Alpha logic:
  1. Track rolling buy/sell imbalance from trade ticks (OFI proxy)
  2. OFI > threshold and price < 0.50: BUY YES (predicting price up -> over)
  3. OFI < -threshold and price > 0.50: SELL (BUY NO equivalent)
  4. Kelly sizing on entry
  5. Hold to settlement (15-minute windows are short)
  6. Safety: halt after max_consecutive_losses, cooldown between entries
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Optional

from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy, StrategyConfig


class BTCOfi15mConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    """Configuration for the BTC OFI 15-minute settlement strategy."""

    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(1)
    ofi_window_ticks: int = 50
    ofi_threshold: float = 0.6
    min_edge_cents: int = 10
    max_entry_price: float = 0.90
    kelly_fraction: float = 0.5
    max_kelly_fraction: float = 0.15
    max_consecutive_losses: int = 3
    cooldown_ticks: int = 20


class BTCOfi15mStrategy(Strategy):
    """BTC OFI 15-minute settlement strategy using trade-tick OFI proxy.

    Tracks rolling buy/sell aggressor imbalance and enters when the
    normalized OFI exceeds the configured threshold.  Consecutive-loss
    halting is safety-critical -- strategy stops emitting signals after
    ``max_consecutive_losses`` losses in a row.
    """

    def __init__(self, config: BTCOfi15mConfig) -> None:
        super().__init__(config)
        self._instrument = None

        # OFI proxy state -- rolling window of +1 (buy) / -1 (sell)
        self._ofi_window: deque[int] = deque(maxlen=config.ofi_window_ticks)
        self._ofi_value: float = 0.0

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
        self._pending = False
        self._in_pos = False
        self._entry_price = None
        self._entry_side = None
        self._consecutive_losses = 0
        self._halted = False
        self._cooldown_remaining = 0
        self._tick_count = 0

    # ------------------------------------------------------------------
    # OFI proxy
    # ------------------------------------------------------------------

    def _update_ofi(self, tick: TradeTick) -> None:
        """Update rolling OFI from trade tick aggressor side."""
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

        # Update OFI proxy
        self._update_ofi(tick)

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

        # Price guard
        if price > self.config.max_entry_price:
            return

        # Edge calculation based on OFI direction
        ofi = self._ofi_value
        price_cents = round(price * 100)

        # BUY YES: OFI bullish (net buy pressure) and price is undervalued
        if ofi > self.config.ofi_threshold and price < 0.50:
            # OFI predicts price going up -> market should settle YES
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
                    f"OFI={ofi:.3f}, FV={fair_cents}c, edge={edge_cents}c"
                )

        # BUY NO (SELL): OFI bearish (net sell pressure) and price is overvalued
        elif ofi < -self.config.ofi_threshold and price > 0.50:
            # OFI predicts price going down -> market should settle NO
            fair_prob_yes = 0.50 - (abs(ofi) - self.config.ofi_threshold) * 0.5
            fair_prob_yes = max(fair_prob_yes, 0.05)
            fair_cents_yes = round(fair_prob_yes * 100)
            edge_cents = price_cents - fair_cents_yes

            if edge_cents >= self.config.min_edge_cents:
                # In Nautilus, selling YES is equivalent to buying NO
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
                    f"OFI={ofi:.3f}, FV_yes={fair_cents_yes}c, edge={edge_cents}c"
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
