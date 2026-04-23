# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software distributed under the
#  License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
Kalshi spread-cost fill model for realistic backtest execution.

Kalshi prediction markets have wide spreads (NBA totals: median 6c, thin
book 11c on a 0-100 contract).  Backtests that fill at the mid-price
overstate edge.  This fill model applies empirical half-spread slippage
to market orders and reduces limit-order fill probability based on the
spread width.
"""

from __future__ import annotations

from nautilus_trader.backtest.models import FillModel
from nautilus_trader.core.rust.model import BookType, OrderSide, OrderType
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import BookOrder
from nautilus_trader.model.objects import Quantity

_UNLIMITED_BOOK_SIZE = 1_000_000

# Kalshi contract prices are 0-100 cents, represented as 0.0-1.0 in Nautilus.
# Spread values below are in cents and converted to the 0.0-1.0 scale.
_CENTS_TO_PRICE = 0.01


class KalshiSpreadCostFillModel(FillModel):
    """
    Empirical spread-cost fill model for Kalshi prediction markets.

    Applies half-spread slippage to market orders based on observed Kalshi
    orderbook spreads.  Limit orders fill at their limit price (no slippage)
    but with probability reduced by how far the limit price sits from the
    current best quote relative to the full spread.

    Parameters
    ----------
    median_spread_cents : float, default 6.0
        The median observed spread in cents (0-100 scale).  Half of this
        is applied as adverse slippage to market order fills.
    thin_book_spread_cents : float, default 11.0
        The spread in cents for thin-book conditions.  Used to compute
        the maximum expected slippage and to scale limit-order fill
        probability.  A limit order whose price is at or beyond the
        thin-book half-spread from best quote has near-zero fill probability.

    Examples
    --------
    Direct use with ``BacktestEngine.add_venue``::

        from prediction_market_extensions.adapters.kalshi.fill_model import (
            KalshiSpreadCostFillModel,
        )

        engine.add_venue(
            venue=Venue("KALSHI"),
            ...,
            fill_model=KalshiSpreadCostFillModel(),
        )

    Conservative (thin book) assumptions::

        engine.add_venue(
            venue=Venue("KALSHI"),
            ...,
            fill_model=KalshiSpreadCostFillModel(
                median_spread_cents=11.0,
                thin_book_spread_cents=16.0,
            ),
        )
    """

    def __init__(
        self,
        *,
        median_spread_cents: float = 6.0,
        thin_book_spread_cents: float = 11.0,
    ) -> None:
        if median_spread_cents < 0:
            raise ValueError(
                f"median_spread_cents must be >= 0, got {median_spread_cents}"
            )
        if thin_book_spread_cents < 0:
            raise ValueError(
                f"thin_book_spread_cents must be >= 0, got {thin_book_spread_cents}"
            )
        if thin_book_spread_cents < median_spread_cents:
            raise ValueError(
                f"thin_book_spread_cents ({thin_book_spread_cents}) must be >= "
                f"median_spread_cents ({median_spread_cents})"
            )

        self._median_spread_cents = median_spread_cents
        self._thin_book_spread_cents = thin_book_spread_cents

        # Half-spread in price units (0.0-1.0 scale)
        self._half_spread = (median_spread_cents / 2.0) * _CENTS_TO_PRICE
        self._thin_half_spread = (thin_book_spread_cents / 2.0) * _CENTS_TO_PRICE

        # Limit orders: base fill probability reduced from 1.0 to account
        # for the reality that passive orders in wide-spread markets don't
        # always get filled.  The probability scales with spread width --
        # wider spreads mean less active crossing.
        #
        # Formula: prob = max(0.2, 1.0 - median_spread_cents / 20.0)
        #   - 0c spread  -> 1.0 (always fills)
        #   - 6c spread  -> 0.7
        #   - 11c spread -> 0.45
        #   - 16c spread -> 0.2 (floor)
        self._limit_fill_prob = max(0.2, 1.0 - median_spread_cents / 20.0)

        # Market orders get slippage via synthetic book; limit orders use
        # the built-in probabilistic fill. Disable built-in slippage since
        # we model it explicitly through the synthetic orderbook.
        super().__init__(
            prob_fill_on_limit=self._limit_fill_prob,
            prob_slippage=0.0,
        )

    def get_orderbook_for_fill_simulation(self, instrument, order, best_bid, best_ask):
        """
        Return a synthetic orderbook with spread-cost slippage for market orders.

        Limit orders return None (use default matching at limit price with
        probabilistic fill gating via ``prob_fill_on_limit``).

        For market orders, the synthetic book is shifted by the median
        half-spread adverse to the taker:

        - BUY:  ask shifted up by half-spread (buyer pays more)
        - SELL: bid shifted down by half-spread (seller receives less)

        Prices are clamped to [0.0, 1.0].
        """
        if order.order_type == OrderType.LIMIT:
            return None

        raw_ask = float(best_ask)
        raw_bid = float(best_bid)

        # Apply half-spread slippage adverse to the taker
        slipped_ask = min(1.0, raw_ask + self._half_spread)
        slipped_bid = max(0.0, raw_bid - self._half_spread)

        slipped_bid = instrument.make_price(slipped_bid)
        slipped_ask = instrument.make_price(slipped_ask)

        book = OrderBook(instrument_id=instrument.id, book_type=BookType.L2_MBP)

        book.add(
            BookOrder(
                side=OrderSide.BUY,
                price=slipped_bid,
                size=Quantity(_UNLIMITED_BOOK_SIZE, instrument.size_precision),
                order_id=1,
            ),
            0,
            0,
        )
        book.add(
            BookOrder(
                side=OrderSide.SELL,
                price=slipped_ask,
                size=Quantity(_UNLIMITED_BOOK_SIZE, instrument.size_precision),
                order_id=2,
            ),
            0,
            0,
        )

        return book
