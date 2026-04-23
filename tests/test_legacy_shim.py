"""Tests for the legacy strategy shim (N3.5)."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Optional

from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue

from strategies.legacy_shim import (
    LegacyShimConfig,
    LegacySignalShim,
    MarketSnapshot,
    QuoteTickLegacyShim,
    SignalResult,
)

INSTRUMENT_ID = InstrumentId(Symbol("PM-TEST-YES"), Venue("POLYMARKET"))


# ---------------------------------------------------------------------------
# Test harnesses — override portfolio/order methods for unit testing
# ---------------------------------------------------------------------------


class _TradeTickHarness(LegacySignalShim):
    """Test harness for LegacySignalShim that avoids Nautilus engine deps."""

    def __init__(self, config: LegacyShimConfig, signal_fn) -> None:
        super().__init__(config, signal_fn)
        self.entries: int = 0
        self.exits: int = 0
        self._position: bool = False

    def _in_position(self) -> bool:
        return self._position

    def _submit_entry(
        self, *, reference_price: float | None = None, visible_size: float | None = None
    ) -> None:
        self.entries += 1
        self._pending = True

    def _submit_exit(self) -> None:
        self.exits += 1
        self._pending = True

    def fill_entry(self, price: float, qty: float = 1.0) -> None:
        self._position = True
        self.on_order_filled(
            SimpleNamespace(order_side=OrderSide.BUY, last_px=price, last_qty=qty)
        )

    def fill_exit(self, price: float, qty: float = 1.0) -> None:
        self._position = False
        self.on_order_filled(
            SimpleNamespace(order_side=OrderSide.SELL, last_px=price, last_qty=qty)
        )


class _QuoteTickHarness(QuoteTickLegacyShim):
    """Test harness for QuoteTickLegacyShim."""

    def __init__(self, config: LegacyShimConfig, signal_fn) -> None:
        super().__init__(config, signal_fn)
        self.entries: int = 0
        self.exits: int = 0
        self._position: bool = False

    def _in_position(self) -> bool:
        return self._position

    def _submit_entry(
        self, *, reference_price: float | None = None, visible_size: float | None = None
    ) -> None:
        self.entries += 1
        self._pending = True

    def _submit_exit(self) -> None:
        self.exits += 1
        self._pending = True

    def fill_entry(self, price: float, qty: float = 1.0) -> None:
        self._position = True
        self.on_order_filled(
            SimpleNamespace(order_side=OrderSide.BUY, last_px=price, last_qty=qty)
        )

    def fill_exit(self, price: float, qty: float = 1.0) -> None:
        self._position = False
        self.on_order_filled(
            SimpleNamespace(order_side=OrderSide.SELL, last_px=price, last_qty=qty)
        )


def _make_trade_tick(price: float) -> SimpleNamespace:
    return SimpleNamespace(price=price)


def _make_quote_tick(bid: float, ask: float) -> SimpleNamespace:
    return SimpleNamespace(bid_price=bid, ask_price=ask, bid_size=10, ask_size=10)


def _default_config(**overrides) -> LegacyShimConfig:
    defaults = dict(instrument_id=INSTRUMENT_ID, trade_size=Decimal(1))
    defaults.update(overrides)
    return LegacyShimConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Test MarketSnapshot construction
# ---------------------------------------------------------------------------


def test_market_snapshot_construction() -> None:
    snap = MarketSnapshot(
        price=0.55,
        bid=0.54,
        ask=0.56,
        spread=0.02,
        tick_count=10,
        in_position=True,
        entry_price=0.50,
        unrealized_pnl=0.05,
    )
    assert snap.price == 0.55
    assert snap.bid == 0.54
    assert snap.ask == 0.56
    assert snap.spread == 0.02
    assert snap.tick_count == 10
    assert snap.in_position is True
    assert snap.entry_price == 0.50
    assert snap.unrealized_pnl == 0.05


# ---------------------------------------------------------------------------
# 2. Test SignalResult creation
# ---------------------------------------------------------------------------


def test_signal_result_creation() -> None:
    sig = SignalResult(action="enter", strength=0.8, reason="cheap")
    assert sig.action == "enter"
    assert sig.strength == 0.8
    assert sig.reason == "cheap"


def test_signal_result_defaults() -> None:
    sig = SignalResult(action="hold")
    assert sig.strength == 1.0
    assert sig.reason == ""


# ---------------------------------------------------------------------------
# 3. Test LegacyShimConfig validation
# ---------------------------------------------------------------------------


def test_legacy_shim_config_defaults() -> None:
    cfg = _default_config()
    assert cfg.instrument_id == INSTRUMENT_ID
    assert cfg.trade_size == Decimal(1)
    assert cfg.take_profit == 0.0
    assert cfg.stop_loss == 0.0


def test_legacy_shim_config_custom_values() -> None:
    cfg = _default_config(
        trade_size=Decimal(5),
        take_profit=0.10,
        stop_loss=0.05,
    )
    assert cfg.trade_size == Decimal(5)
    assert cfg.take_profit == 0.10
    assert cfg.stop_loss == 0.05


# ---------------------------------------------------------------------------
# 4. Test signal_fn is called with correct MarketSnapshot fields
# ---------------------------------------------------------------------------


def test_signal_fn_receives_correct_snapshot() -> None:
    received_snapshots = []

    def capture_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        received_snapshots.append(snap)
        return None

    strategy = _TradeTickHarness(_default_config(), capture_fn)
    strategy.on_trade_tick(_make_trade_tick(0.45))

    assert len(received_snapshots) == 1
    s = received_snapshots[0]
    assert s.price == 0.45
    assert s.bid == 0.45  # trade tick -> bid=ask=price
    assert s.ask == 0.45
    assert s.spread == 0.0
    assert s.tick_count == 1
    assert s.in_position is False
    assert s.entry_price == 0.0
    assert s.unrealized_pnl == 0.0


def test_signal_fn_receives_correct_snapshot_when_in_position() -> None:
    received_snapshots = []

    def capture_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        received_snapshots.append(snap)
        return None

    strategy = _TradeTickHarness(_default_config(), capture_fn)

    # Simulate entry
    strategy.on_trade_tick(_make_trade_tick(0.40))
    strategy.fill_entry(0.40)

    # Next tick while in position
    strategy.on_trade_tick(_make_trade_tick(0.45))

    assert len(received_snapshots) == 2
    s = received_snapshots[1]
    assert s.in_position is True
    assert s.entry_price == 0.40
    assert abs(s.unrealized_pnl - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# 5. Test enter/exit/hold signal handling
# ---------------------------------------------------------------------------


def test_enter_signal_submits_entry() -> None:
    def enter_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        return SignalResult(action="enter")

    strategy = _TradeTickHarness(_default_config(), enter_fn)
    strategy.on_trade_tick(_make_trade_tick(0.50))

    assert strategy.entries == 1
    assert strategy.exits == 0


def test_exit_signal_submits_exit_when_in_position() -> None:
    def exit_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        return SignalResult(action="exit")

    strategy = _TradeTickHarness(_default_config(), exit_fn)
    # Enter first
    strategy.on_trade_tick(_make_trade_tick(0.50))  # no entry because exit_fn returns exit
    # Manually set position
    strategy._position = True
    strategy._pending = False
    strategy.on_trade_tick(_make_trade_tick(0.55))

    assert strategy.exits == 1


def test_exit_signal_ignored_when_flat() -> None:
    def exit_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        return SignalResult(action="exit")

    strategy = _TradeTickHarness(_default_config(), exit_fn)
    strategy.on_trade_tick(_make_trade_tick(0.50))

    assert strategy.entries == 0
    assert strategy.exits == 0


def test_hold_signal_does_nothing() -> None:
    def hold_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        return SignalResult(action="hold")

    strategy = _TradeTickHarness(_default_config(), hold_fn)
    strategy.on_trade_tick(_make_trade_tick(0.50))

    assert strategy.entries == 0
    assert strategy.exits == 0


def test_none_signal_does_nothing() -> None:
    def none_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        return None

    strategy = _TradeTickHarness(_default_config(), none_fn)
    strategy.on_trade_tick(_make_trade_tick(0.50))

    assert strategy.entries == 0
    assert strategy.exits == 0


def test_enter_signal_ignored_when_already_in_position() -> None:
    def enter_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        return SignalResult(action="enter")

    strategy = _TradeTickHarness(_default_config(), enter_fn)
    strategy.on_trade_tick(_make_trade_tick(0.50))
    assert strategy.entries == 1
    strategy.fill_entry(0.50)

    strategy.on_trade_tick(_make_trade_tick(0.55))
    assert strategy.entries == 1  # not incremented


# ---------------------------------------------------------------------------
# 6. Test risk exit (take_profit, stop_loss)
# ---------------------------------------------------------------------------


def test_take_profit_triggers_exit() -> None:
    def hold_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        return SignalResult(action="hold")

    config = _default_config(take_profit=0.10, stop_loss=0.0)
    strategy = _TradeTickHarness(config, hold_fn)

    strategy._position = True
    strategy._entry_price = 0.40
    strategy._entry_qty_sum = 1.0
    strategy._entry_cost_sum = 0.40

    strategy.on_trade_tick(_make_trade_tick(0.50))
    assert strategy.exits == 1


def test_stop_loss_triggers_exit() -> None:
    def hold_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        return SignalResult(action="hold")

    config = _default_config(take_profit=0.0, stop_loss=0.05)
    strategy = _TradeTickHarness(config, hold_fn)

    strategy._position = True
    strategy._entry_price = 0.40
    strategy._entry_qty_sum = 1.0
    strategy._entry_cost_sum = 0.40

    strategy.on_trade_tick(_make_trade_tick(0.35))
    assert strategy.exits == 1


def test_no_risk_exit_when_within_bounds() -> None:
    def hold_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        return SignalResult(action="hold")

    config = _default_config(take_profit=0.10, stop_loss=0.05)
    strategy = _TradeTickHarness(config, hold_fn)

    strategy._position = True
    strategy._entry_price = 0.40
    strategy._entry_qty_sum = 1.0
    strategy._entry_cost_sum = 0.40

    strategy.on_trade_tick(_make_trade_tick(0.42))
    assert strategy.exits == 0


def test_no_risk_exit_when_disabled() -> None:
    def hold_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        return SignalResult(action="hold")

    config = _default_config(take_profit=0.0, stop_loss=0.0)
    strategy = _TradeTickHarness(config, hold_fn)

    strategy._position = True
    strategy._entry_price = 0.40
    strategy._entry_qty_sum = 1.0
    strategy._entry_cost_sum = 0.40

    strategy.on_trade_tick(_make_trade_tick(0.90))
    assert strategy.exits == 0


# ---------------------------------------------------------------------------
# 7. Test QuoteTickLegacyShim with bid/ask data
# ---------------------------------------------------------------------------


def test_quote_tick_shim_uses_real_bbo() -> None:
    received_snapshots = []

    def capture_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        received_snapshots.append(snap)
        return None

    strategy = _QuoteTickHarness(_default_config(), capture_fn)
    strategy.on_quote_tick(_make_quote_tick(bid=0.48, ask=0.52))

    assert len(received_snapshots) == 1
    s = received_snapshots[0]
    assert s.bid == 0.48
    assert s.ask == 0.52
    assert abs(s.spread - 0.04) < 1e-9
    assert abs(s.price - 0.50) < 1e-9


def test_quote_tick_shim_enter_signal() -> None:
    def enter_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        if snap.price < 0.40:
            return SignalResult(action="enter")
        return None

    strategy = _QuoteTickHarness(_default_config(), enter_fn)
    strategy.on_quote_tick(_make_quote_tick(bid=0.34, ask=0.38))

    assert strategy.entries == 1


def test_quote_tick_shim_exit_signal() -> None:
    def exit_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        return SignalResult(action="exit")

    strategy = _QuoteTickHarness(_default_config(), exit_fn)
    strategy._position = True
    strategy._entry_price = 0.30
    strategy._entry_qty_sum = 1.0
    strategy._entry_cost_sum = 0.30

    strategy.on_quote_tick(_make_quote_tick(bid=0.48, ask=0.52))
    assert strategy.exits == 1


# ---------------------------------------------------------------------------
# 8. Test "buy below 0.40" end-to-end signal function
# ---------------------------------------------------------------------------


def test_buy_below_040_signal_function() -> None:
    def buy_below_040(snap: MarketSnapshot) -> Optional[SignalResult]:
        if not snap.in_position and snap.price < 0.40:
            return SignalResult(action="enter", reason="cheap")
        if snap.in_position and snap.price >= 0.60:
            return SignalResult(action="exit", reason="target")
        return None

    strategy = _TradeTickHarness(_default_config(), buy_below_040)

    # Price above threshold — no entry
    strategy.on_trade_tick(_make_trade_tick(0.50))
    assert strategy.entries == 0

    # Price below threshold — entry
    strategy.on_trade_tick(_make_trade_tick(0.35))
    assert strategy.entries == 1
    strategy.fill_entry(0.35)

    # Price below exit target — hold
    strategy.on_trade_tick(_make_trade_tick(0.45))
    assert strategy.exits == 0

    # Price hits exit target — exit
    strategy.on_trade_tick(_make_trade_tick(0.60))
    assert strategy.exits == 1
    strategy.fill_exit(0.60)

    # Re-enter on another dip
    strategy.on_trade_tick(_make_trade_tick(0.30))
    assert strategy.entries == 2


# ---------------------------------------------------------------------------
# 9. Test that shim ignores signals when pending order exists
# ---------------------------------------------------------------------------


def test_shim_ignores_signals_when_pending() -> None:
    call_count = 0

    def counting_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        nonlocal call_count
        call_count += 1
        return SignalResult(action="enter")

    strategy = _TradeTickHarness(_default_config(), counting_fn)

    # First tick triggers entry — sets pending
    strategy.on_trade_tick(_make_trade_tick(0.50))
    assert strategy.entries == 1
    assert strategy._pending is True
    first_call_count = call_count

    # Second tick while pending — signal_fn should NOT be called
    strategy.on_trade_tick(_make_trade_tick(0.50))
    assert strategy.entries == 1  # no additional entry
    assert call_count == first_call_count  # fn not called


def test_quote_tick_shim_ignores_signals_when_pending() -> None:
    call_count = 0

    def counting_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        nonlocal call_count
        call_count += 1
        return SignalResult(action="enter")

    strategy = _QuoteTickHarness(_default_config(), counting_fn)

    strategy.on_quote_tick(_make_quote_tick(bid=0.48, ask=0.52))
    assert strategy.entries == 1
    assert strategy._pending is True
    first_call_count = call_count

    strategy.on_quote_tick(_make_quote_tick(bid=0.48, ask=0.52))
    assert strategy.entries == 1
    assert call_count == first_call_count


# ---------------------------------------------------------------------------
# Additional: tick_count increments correctly
# ---------------------------------------------------------------------------


def test_tick_count_increments() -> None:
    received_counts = []

    def capture_fn(snap: MarketSnapshot) -> Optional[SignalResult]:
        received_counts.append(snap.tick_count)
        return None

    strategy = _TradeTickHarness(_default_config(), capture_fn)
    strategy.on_trade_tick(_make_trade_tick(0.50))
    strategy.on_trade_tick(_make_trade_tick(0.51))
    strategy.on_trade_tick(_make_trade_tick(0.52))

    assert received_counts == [1, 2, 3]
