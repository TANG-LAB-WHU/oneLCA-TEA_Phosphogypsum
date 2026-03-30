"""
Currency Conversion Module

Currency exchange rates and inflation adjustment for TEA calculations.
"""

from dataclasses import dataclass

# Exchange rates (relative to USD, 2024 baseline)
EXCHANGE_RATES_2024 = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "CNY": 7.25,
    "JPY": 149.0,
    "INR": 83.0,
    "BRL": 4.95,
    "RUB": 90.0,
    "KRW": 1320.0,
    "AUD": 1.53,
    "CAD": 1.36,
}

# Annual inflation rates by currency (2020-2024 average)
INFLATION_RATES = {
    "USD": 0.035,  # 3.5%
    "EUR": 0.032,
    "GBP": 0.040,
    "CNY": 0.020,
    "JPY": 0.015,
    "INR": 0.055,
    "BRL": 0.060,
}

# Regional cost adjustment factors (relative to US)
REGIONAL_FACTORS = {
    "US": 1.0,
    "EU": 1.15,
    "China": 0.65,
    "India": 0.45,
    "Brazil": 0.55,
    "Japan": 1.25,
    "Australia": 1.20,
}


def get_exchange_rate(from_currency: str, to_currency: str, year: int = 2024) -> float:
    """
    Get exchange rate between two currencies.

    Args:
        from_currency: Source currency code (e.g., 'USD')
        to_currency: Target currency code (e.g., 'EUR')
        year: Year for rate (currently only 2024 supported)

    Returns:
        Exchange rate (to_currency per from_currency)
    """
    if from_currency not in EXCHANGE_RATES_2024:
        raise ValueError(f"Unknown currency: {from_currency}")
    if to_currency not in EXCHANGE_RATES_2024:
        raise ValueError(f"Unknown currency: {to_currency}")

    # Convert via USD
    usd_per_from = 1.0 / EXCHANGE_RATES_2024[from_currency]
    to_per_usd = EXCHANGE_RATES_2024[to_currency]

    return usd_per_from * to_per_usd


def convert_currency(
    amount: float, from_currency: str, to_currency: str, year: int = 2024
) -> float:
    """
    Convert amount between currencies.

    Args:
        amount: Amount in source currency
        from_currency: Source currency code
        to_currency: Target currency code
        year: Year for exchange rate

    Returns:
        Amount in target currency
    """
    rate = get_exchange_rate(from_currency, to_currency, year)
    return amount * rate


def adjust_inflation(amount: float, from_year: int, to_year: int, currency: str = "USD") -> float:
    """
    Adjust amount for inflation between years.

    Args:
        amount: Original amount
        from_year: Original year
        to_year: Target year
        currency: Currency for inflation rate

    Returns:
        Inflation-adjusted amount
    """
    if currency not in INFLATION_RATES:
        # Use USD as default
        inflation_rate = INFLATION_RATES["USD"]
    else:
        inflation_rate = INFLATION_RATES[currency]

    years_diff = to_year - from_year
    adjustment_factor = (1 + inflation_rate) ** years_diff

    return amount * adjustment_factor


def get_regional_factor(region: str) -> float:
    """Get regional cost adjustment factor."""
    return REGIONAL_FACTORS.get(region, 1.0)


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency string."""
    if currency == "USD":
        return f"${amount:,.2f}"
    elif currency == "CNY":
        return f"¥{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


@dataclass
class CurrencyConverter:
    """
    Currency converter with caching and regional adjustments.
    """

    base_currency: str = "USD"
    base_year: int = 2024
    target_region: str = "US"

    def convert(self, amount: float, from_currency: str = None, from_year: int = None) -> float:
        """
        Convert and adjust amount to base currency and year.

        Args:
            amount: Original amount
            from_currency: Source currency (default: base_currency)
            from_year: Source year (default: base_year)

        Returns:
            Converted and adjusted amount
        """
        from_currency = from_currency or self.base_currency
        from_year = from_year or self.base_year

        # Currency conversion
        if from_currency != self.base_currency:
            amount = convert_currency(amount, from_currency, self.base_currency)

        # Inflation adjustment
        if from_year != self.base_year:
            amount = adjust_inflation(amount, from_year, self.base_year, self.base_currency)

        return amount

    def apply_regional_factor(self, amount: float, region: str = None) -> float:
        """Apply regional cost adjustment."""
        region = region or self.target_region
        return amount * get_regional_factor(region)
