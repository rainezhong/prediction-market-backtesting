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
#  Derived from NautilusTrader prediction-market example code.
#  See the repository NOTICE file for provenance and licensing scope.
#

"""Custom data types for prediction market strategies."""

from __future__ import annotations

from prediction_market_extensions.data.custom_types import (
    BasketballContextData,
    F1TelemetryData,
    basketball_data_type,
    f1_data_type,
)

__all__ = [
    "BasketballContextData",
    "F1TelemetryData",
    "basketball_data_type",
    "f1_data_type",
]
