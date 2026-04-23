from __future__ import annotations

import types
from decimal import Decimal

import pytest
from nautilus_trader.core.rust.model import OrderType
from nautilus_trader.model.enums import OrderSide as OrderSideEnum
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Price

from prediction_market_extensions.adapters.kalshi.fill_model import (
    KalshiSpreadCostFillModel,
)


class _StubInstrument:
    """Minimal instrument stub for fill model tests."""

    def __init__(self, venue: str = "KALSHI", price_increment: float = 0.01) -> None:
        self.id = InstrumentId(Symbol("TEST-YES"), Venue(venue))
        self.price_increment = Decimal(str(price_increment))
        self.size_precision = 0

    def make_price(self, value: float) -> Price:
        return Price.from_str(f"{value:.4f}")


def _make_order(side: OrderSideEnum, order_type=OrderType.MARKET):
    return types.SimpleNamespace(side=side, order_type=order_type)


# ── Construction ──────────────────────────────────────────────────────────


def test_default_params() -> None:
    model = KalshiSpreadCostFillModel()
    assert model._median_spread_cents == 6.0
    assert model._thin_book_spread_cents == 11.0
    assert model._half_spread == pytest.approx(0.03)  # 6/2 * 0.01
    assert model._thin_half_spread == pytest.approx(0.055)  # 11/2 * 0.01


def test_custom_spreads() -> None:
    model = KalshiSpreadCostFillModel(
        median_spread_cents=10.0, thin_book_spread_cents=15.0
    )
    assert model._half_spread == pytest.approx(0.05)
    assert model._thin_half_spread == pytest.approx(0.075)


def test_zero_spread_valid() -> None:
    model = KalshiSpreadCostFillModel(
        median_spread_cents=0.0, thin_book_spread_cents=0.0
    )
    assert model._half_spread == 0.0


def test_negative_median_raises() -> None:
    with pytest.raises(ValueError, match="median_spread_cents must be >= 0"):
        KalshiSpreadCostFillModel(median_spread_cents=-1.0)


def test_negative_thin_raises() -> None:
    with pytest.raises(ValueError, match="thin_book_spread_cents must be >= 0"):
        KalshiSpreadCostFillModel(thin_book_spread_cents=-1.0)


def test_thin_less_than_median_raises() -> None:
    with pytest.raises(ValueError, match="thin_book_spread_cents.*must be >="):
        KalshiSpreadCostFillModel(
            median_spread_cents=10.0, thin_book_spread_cents=5.0
        )


# ── Market order slippage ────────────────────────────────────────────────


def test_buy_market_ask_shifted_up() -> None:
    """BUY market order: ask shifts up by half-spread (3c = 0.03)."""
    model = KalshiSpreadCostFillModel()
    instrument = _StubInstrument()
    book = model.get_orderbook_for_fill_simulation(
        instrument,
        _make_order(OrderSideEnum.BUY),
        best_bid=Price.from_str("0.5000"),
        best_ask=Price.from_str("0.5100"),
    )
    assert book is not None
    # ask should be 0.51 + 0.03 = 0.54
    assert float(book.best_ask_price()) == pytest.approx(0.54)


def test_sell_market_bid_shifted_down() -> None:
    """SELL market order: bid shifts down by half-spread (3c = 0.03)."""
    model = KalshiSpreadCostFillModel()
    instrument = _StubInstrument()
    book = model.get_orderbook_for_fill_simulation(
        instrument,
        _make_order(OrderSideEnum.SELL),
        best_bid=Price.from_str("0.5000"),
        best_ask=Price.from_str("0.5100"),
    )
    assert book is not None
    # bid should be 0.50 - 0.03 = 0.47
    assert float(book.best_bid_price()) == pytest.approx(0.47)


def test_buy_clamped_at_one() -> None:
    """Price should not exceed 1.0."""
    model = KalshiSpreadCostFillModel(median_spread_cents=20.0, thin_book_spread_cents=20.0)
    instrument = _StubInstrument()
    book = model.get_orderbook_for_fill_simulation(
        instrument,
        _make_order(OrderSideEnum.BUY),
        best_bid=Price.from_str("0.9500"),
        best_ask=Price.from_str("0.9500"),
    )
    assert book is not None
    assert float(book.best_ask_price()) == 1.0


def test_sell_clamped_at_zero() -> None:
    """Price should not go below 0.0."""
    model = KalshiSpreadCostFillModel(median_spread_cents=20.0, thin_book_spread_cents=20.0)
    instrument = _StubInstrument()
    book = model.get_orderbook_for_fill_simulation(
        instrument,
        _make_order(OrderSideEnum.SELL),
        best_bid=Price.from_str("0.0500"),
        best_ask=Price.from_str("0.0500"),
    )
    assert book is not None
    assert float(book.best_bid_price()) == 0.0


def test_symmetric_slippage() -> None:
    """Both sides of the book shift symmetrically."""
    model = KalshiSpreadCostFillModel()
    instrument = _StubInstrument()
    book = model.get_orderbook_for_fill_simulation(
        instrument,
        _make_order(OrderSideEnum.BUY),
        best_bid=Price.from_str("0.5000"),
        best_ask=Price.from_str("0.5000"),
    )
    assert book is not None
    # bid: 0.50 - 0.03 = 0.47, ask: 0.50 + 0.03 = 0.53
    assert float(book.best_bid_price()) == pytest.approx(0.47)
    assert float(book.best_ask_price()) == pytest.approx(0.53)


def test_zero_spread_no_slippage() -> None:
    """Zero spread config produces no slippage."""
    model = KalshiSpreadCostFillModel(
        median_spread_cents=0.0, thin_book_spread_cents=0.0
    )
    instrument = _StubInstrument()
    book = model.get_orderbook_for_fill_simulation(
        instrument,
        _make_order(OrderSideEnum.BUY),
        best_bid=Price.from_str("0.5000"),
        best_ask=Price.from_str("0.5100"),
    )
    assert book is not None
    assert float(book.best_ask_price()) == pytest.approx(0.51)
    assert float(book.best_bid_price()) == pytest.approx(0.50)


def test_thin_book_spread_larger_slippage() -> None:
    """Using thin-book-level median produces more slippage."""
    model = KalshiSpreadCostFillModel(
        median_spread_cents=11.0, thin_book_spread_cents=16.0
    )
    instrument = _StubInstrument()
    book = model.get_orderbook_for_fill_simulation(
        instrument,
        _make_order(OrderSideEnum.BUY),
        best_bid=Price.from_str("0.5000"),
        best_ask=Price.from_str("0.5100"),
    )
    assert book is not None
    # half-spread = 11/2 * 0.01 = 0.055
    # ask = 0.51 + 0.055 = 0.565
    assert float(book.best_ask_price()) == pytest.approx(0.565)


# ── Limit orders ─────────────────────────────────────────────────────────


def test_limit_order_returns_none() -> None:
    """Limit orders use default matching (no synthetic book)."""
    model = KalshiSpreadCostFillModel()
    instrument = _StubInstrument()
    book = model.get_orderbook_for_fill_simulation(
        instrument,
        _make_order(OrderSideEnum.BUY, OrderType.LIMIT),
        best_bid=Price.from_str("0.5000"),
        best_ask=Price.from_str("0.5100"),
    )
    assert book is None


def test_limit_fill_probability_default() -> None:
    """Default 6c spread -> prob = 1.0 - 6/20 = 0.7."""
    model = KalshiSpreadCostFillModel()
    assert model.prob_fill_on_limit == pytest.approx(0.7)


def test_limit_fill_probability_zero_spread() -> None:
    """0c spread -> prob = 1.0."""
    model = KalshiSpreadCostFillModel(
        median_spread_cents=0.0, thin_book_spread_cents=0.0
    )
    assert model.prob_fill_on_limit == pytest.approx(1.0)


def test_limit_fill_probability_wide_spread() -> None:
    """16c spread -> prob = 0.2 (floor)."""
    model = KalshiSpreadCostFillModel(
        median_spread_cents=16.0, thin_book_spread_cents=20.0
    )
    assert model.prob_fill_on_limit == pytest.approx(0.2)


def test_limit_fill_probability_very_wide_spread() -> None:
    """25c spread -> prob = 0.2 (floor, not negative)."""
    model = KalshiSpreadCostFillModel(
        median_spread_cents=25.0, thin_book_spread_cents=30.0
    )
    assert model.prob_fill_on_limit == pytest.approx(0.2)


def test_limit_fill_probability_thin_book() -> None:
    """11c spread -> prob = 1.0 - 11/20 = 0.45."""
    model = KalshiSpreadCostFillModel(
        median_spread_cents=11.0, thin_book_spread_cents=16.0
    )
    assert model.prob_fill_on_limit == pytest.approx(0.45)


# ── Edge cases ───────────────────────────────────────────────────────────


def test_bid_ask_book_not_inverted() -> None:
    """Synthetic book bid should always be <= ask."""
    model = KalshiSpreadCostFillModel()
    instrument = _StubInstrument()
    book = model.get_orderbook_for_fill_simulation(
        instrument,
        _make_order(OrderSideEnum.BUY),
        best_bid=Price.from_str("0.5000"),
        best_ask=Price.from_str("0.5100"),
    )
    assert book is not None
    assert float(book.best_bid_price()) < float(book.best_ask_price())


def test_at_extremes_low_price() -> None:
    """Low price near 0 should not produce negative bid."""
    model = KalshiSpreadCostFillModel()
    instrument = _StubInstrument()
    book = model.get_orderbook_for_fill_simulation(
        instrument,
        _make_order(OrderSideEnum.SELL),
        best_bid=Price.from_str("0.0200"),
        best_ask=Price.from_str("0.0300"),
    )
    assert book is not None
    assert float(book.best_bid_price()) >= 0.0


def test_at_extremes_high_price() -> None:
    """High price near 1 should not produce ask > 1."""
    model = KalshiSpreadCostFillModel()
    instrument = _StubInstrument()
    book = model.get_orderbook_for_fill_simulation(
        instrument,
        _make_order(OrderSideEnum.BUY),
        best_bid=Price.from_str("0.9800"),
        best_ask=Price.from_str("0.9900"),
    )
    assert book is not None
    assert float(book.best_ask_price()) <= 1.0
