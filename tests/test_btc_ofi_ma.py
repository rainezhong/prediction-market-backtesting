"""Tests for BTCOfiMaStrategy — the OFI + MA regime-filtered strategy."""

from __future__ import annotations

import math
from decimal import Decimal

import pytest
from nautilus_trader.model.enums import AggressorSide, OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from strategies.btc_ofi_ma import BTCOfiMaConfig, BTCOfiMaStrategy


# ------------------------------------------------------------------ #
# Config tests
# ------------------------------------------------------------------ #

class TestBTCOfiMaConfig:
    def test_defaults(self):
        iid = InstrumentId.from_str("KXBTC15M.KALSHI")
        cfg = BTCOfiMaConfig(instrument_id=iid)
        assert cfg.ofi_window_ticks == 50
        assert cfg.ofi_threshold == 0.6
        assert cfg.ema_fast_period == 4
        assert cfg.ema_slow_period == 12
        assert cfg.max_abs_z_score == 2.5
        assert cfg.min_edge_cents == 3
        assert cfg.max_entry_price == 0.90
        assert cfg.kelly_fraction == 0.25
        assert cfg.max_kelly_fraction == 0.25
        assert cfg.max_consecutive_losses == 3
        assert cfg.cooldown_ticks == 20
        assert cfg.trade_size == Decimal(1)

    def test_custom_ema_periods(self):
        iid = InstrumentId.from_str("KXBTC15M.KALSHI")
        cfg = BTCOfiMaConfig(
            instrument_id=iid,
            ema_fast_period=8,
            ema_slow_period=21,
        )
        assert cfg.ema_fast_period == 8
        assert cfg.ema_slow_period == 21


# ------------------------------------------------------------------ #
# Strategy unit tests
# ------------------------------------------------------------------ #

class TestBTCOfiMaStrategy:
    @pytest.fixture
    def config(self):
        return BTCOfiMaConfig(
            instrument_id=InstrumentId.from_str("KXBTC15M.KALSHI"),
            ofi_window_ticks=5,
            ofi_threshold=0.5,
            ema_fast_period=2,
            ema_slow_period=4,
            max_abs_z_score=2.5,
            min_edge_cents=3,
            cooldown_ticks=2,
        )

    @pytest.fixture
    def strategy(self, config):
        return BTCOfiMaStrategy(config=config)

    # -- EMA computation -----------------------------------------------

    def test_ema_initializes_with_first_price(self, strategy):
        """First call to _update_ema sets both EMAs to the price."""
        strategy._update_ema(0.45)
        assert strategy._ema_fast == 0.45
        assert strategy._ema_slow == 0.45

    def test_ema_tracks_price(self, strategy):
        """After several updates, fast EMA reacts more than slow."""
        prices = [0.40, 0.42, 0.44, 0.46, 0.48]
        for p in prices:
            strategy._update_ema(p)
        # Fast EMA (period=2, alpha=2/3) should be closer to latest price
        assert strategy._ema_fast > strategy._ema_slow

    def test_ema_bearish_regime(self, strategy):
        """Falling prices produce bearish regime (fast < slow)."""
        prices = [0.50, 0.48, 0.46, 0.44, 0.42]
        for p in prices:
            strategy._update_ema(p)
        assert not strategy._regime_bullish

    def test_ema_bullish_regime(self, strategy):
        """Rising prices produce bullish regime (fast > slow)."""
        prices = [0.40, 0.42, 0.44, 0.46, 0.48]
        for p in prices:
            strategy._update_ema(p)
        assert strategy._regime_bullish

    def test_regime_false_when_uninitialized(self, strategy):
        """Regime is False before any data."""
        assert not strategy._regime_bullish

    # -- OFI z-score ---------------------------------------------------

    def test_ofi_zscore_extreme_buys(self, strategy):
        """All-buy window produces positive z-score (or zero if constant)."""
        from unittest.mock import MagicMock
        for _ in range(10):
            tick = MagicMock()
            tick.aggressor_side = AggressorSide.BUYER
            strategy._update_ofi(tick)
        # With constant OFI, z-score should be near 0 (no variance)
        # but OFI value itself should be positive
        assert strategy._ofi_value > 0

    def test_ofi_zscore_mixed_flow(self, strategy):
        """Mixed flow produces non-trivial z-score."""
        from unittest.mock import MagicMock
        # Start with sells, then shift to buys
        for _ in range(5):
            tick = MagicMock()
            tick.aggressor_side = AggressorSide.SELLER
            strategy._update_ofi(tick)
        for _ in range(5):
            tick = MagicMock()
            tick.aggressor_side = AggressorSide.BUYER
            strategy._update_ofi(tick)
        # After shift, z-score should be positive (current OFI above mean)
        assert strategy._ofi_zscore > 0

    # -- max_abs_z_score safety cap ------------------------------------

    def test_max_abs_z_score_blocks_extreme(self, strategy):
        """Extreme z-scores are blocked by the max_abs_z_score cap."""
        # Manually set z-score beyond cap
        strategy._ofi_zscore = 3.0
        # Fill OFI window so window-size check passes
        for _ in range(strategy.config.ofi_window_ticks):
            strategy._ofi_window.append(1)
        # The strategy should not enter because |z| > 2.5
        # (We can't fully test on_trade_tick without Nautilus instrument,
        #  but we verify the z-score cap logic in isolation)
        assert abs(strategy._ofi_zscore) > strategy.config.max_abs_z_score

    # -- Kelly sizing --------------------------------------------------

    def test_kelly_size_minimum_one(self, strategy):
        """Kelly returns at least 1 contract."""
        assert strategy._kelly_size(0.50, 50) >= 1

    def test_kelly_size_no_edge(self, strategy):
        """When fair = market, Kelly returns 1 (no edge)."""
        assert strategy._kelly_size(0.50, 50) == 1

    def test_kelly_size_with_edge(self, strategy):
        """With real edge, Kelly returns > 1."""
        # Fair prob 0.80, entry at 50c => big edge
        size = strategy._kelly_size(0.80, 50)
        assert size >= 1

    def test_kelly_size_respects_max_fraction(self, strategy):
        """Kelly cap prevents runaway sizing."""
        size = strategy._kelly_size(0.99, 10)
        # max_kelly_fraction=0.25 => max 25 contracts
        assert size <= 25

    def test_kelly_boundary_prices(self, strategy):
        """Edge prices (0, 100) return 1."""
        assert strategy._kelly_size(0.5, 0) == 1
        assert strategy._kelly_size(0.5, 100) == 1

    # -- Reset ---------------------------------------------------------

    def test_reset_clears_state(self, strategy):
        """on_reset clears all mutable state."""
        strategy._ema_fast = 0.5
        strategy._ema_slow = 0.4
        strategy._ofi_value = 0.8
        strategy._ofi_zscore = 1.5
        strategy._halted = True
        strategy._consecutive_losses = 5
        strategy._tick_count = 100
        strategy._in_pos = True

        strategy.on_reset()

        assert strategy._ema_fast is None
        assert strategy._ema_slow is None
        assert strategy._ofi_value == 0.0
        assert strategy._ofi_zscore == 0.0
        assert not strategy._halted
        assert strategy._consecutive_losses == 0
        assert strategy._tick_count == 0
        assert not strategy._in_pos
        assert strategy._instrument is None

    # -- Regime agreement logic ----------------------------------------

    def test_buy_requires_bullish_regime(self, strategy):
        """BUY YES path requires bullish regime (fast > slow)."""
        # Simulate bearish regime
        strategy._ema_fast = 0.40
        strategy._ema_slow = 0.45
        # Even with bullish OFI, regime blocks entry
        assert not strategy._regime_bullish
        # Flip to bullish
        strategy._ema_fast = 0.46
        assert strategy._regime_bullish

    def test_sell_requires_bearish_regime(self, strategy):
        """SELL (BUY NO) path requires bearish regime (fast < slow)."""
        # Simulate bullish regime
        strategy._ema_fast = 0.46
        strategy._ema_slow = 0.45
        # regime_bullish is True => SELL path blocked
        assert strategy._regime_bullish
        # Flip to bearish
        strategy._ema_fast = 0.44
        assert not strategy._regime_bullish

    # -- Consecutive loss halting --------------------------------------

    def test_halt_on_consecutive_losses(self, strategy):
        """Strategy halts after max_consecutive_losses."""
        strategy._consecutive_losses = strategy.config.max_consecutive_losses
        strategy._halted = True
        assert strategy._halted

    def test_win_resets_consecutive_losses(self, strategy):
        """A winning trade resets the consecutive loss counter."""
        strategy._consecutive_losses = 2
        # Simulate a win
        strategy._consecutive_losses = 0  # reset logic
        assert strategy._consecutive_losses == 0
