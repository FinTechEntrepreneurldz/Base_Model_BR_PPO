"""
Check whether today is a NYSE trading day.
Exits with code 0 (is trading day) or 1 (not a trading day).

Usage:
    python scripts/market_check.py        # exits 0 or 1
    python scripts/market_check.py --info # prints details, still exits 0 or 1
"""

import sys
import datetime

def is_trading_day(date=None):
    """
    Returns True if `date` (default: today ET) is a NYSE trading day.
    Uses pandas_market_calendars if available, otherwise a simple
    weekday + hardcoded holiday check.
    """
    import pytz

    et  = pytz.timezone("America/New_York")
    now = datetime.datetime.now(et)
    d   = date or now.date()

    # Weekend
    if d.weekday() >= 5:
        return False, "Weekend"

    # Try pandas_market_calendars first (most accurate)
    try:
        import pandas_market_calendars as mcal
        import pandas as pd
        nyse     = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=str(d), end_date=str(d))
        if schedule.empty:
            return False, "NYSE holiday"
        return True, "Trading day"
    except ImportError:
        pass

    # Fallback: hardcoded US federal market holidays for 2024-2026
    HOLIDAYS = {
        # 2024
        datetime.date(2024,  1,  1),  # New Year's Day
        datetime.date(2024,  1, 15),  # MLK Day
        datetime.date(2024,  2, 19),  # Presidents' Day
        datetime.date(2024,  3, 29),  # Good Friday
        datetime.date(2024,  5, 27),  # Memorial Day
        datetime.date(2024,  6, 19),  # Juneteenth
        datetime.date(2024,  7,  4),  # Independence Day
        datetime.date(2024,  9,  2),  # Labor Day
        datetime.date(2024, 11, 28),  # Thanksgiving
        datetime.date(2024, 12, 25),  # Christmas
        # 2025
        datetime.date(2025,  1,  1),  # New Year's Day
        datetime.date(2025,  1, 20),  # MLK Day
        datetime.date(2025,  2, 17),  # Presidents' Day
        datetime.date(2025,  4, 18),  # Good Friday
        datetime.date(2025,  5, 26),  # Memorial Day
        datetime.date(2025,  6, 19),  # Juneteenth
        datetime.date(2025,  7,  4),  # Independence Day
        datetime.date(2025,  9,  1),  # Labor Day
        datetime.date(2025, 11, 27),  # Thanksgiving
        datetime.date(2025, 12, 25),  # Christmas
        # 2026
        datetime.date(2026,  1,  1),  # New Year's Day
        datetime.date(2026,  1, 19),  # MLK Day
        datetime.date(2026,  2, 16),  # Presidents' Day
        datetime.date(2026,  4,  3),  # Good Friday
        datetime.date(2026,  5, 25),  # Memorial Day
        datetime.date(2026,  6, 19),  # Juneteenth
        datetime.date(2026,  7,  3),  # Independence Day (observed)
        datetime.date(2026,  9,  7),  # Labor Day
        datetime.date(2026, 11, 26),  # Thanksgiving
        datetime.date(2026, 12, 25),  # Christmas
    }

    if d in HOLIDAYS:
        return False, "NYSE holiday (hardcoded)"

    return True, "Trading day (weekday, not holiday)"


if __name__ == "__main__":
    verbose = "--info" in sys.argv

    trading, reason = is_trading_day()

    if verbose:
        import pytz, datetime
        et  = pytz.timezone("America/New_York")
        now = datetime.datetime.now(et)
        print(f"Date (ET)   : {now.strftime('%Y-%m-%d %H:%M %Z')}")
        print(f"Trading day : {trading}")
        print(f"Reason      : {reason}")

    sys.exit(0 if trading else 1)
