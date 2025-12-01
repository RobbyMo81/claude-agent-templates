"""Time utilities for trading applications - Eastern Time Standard"""

from typing import Optional, List, Union, Tuple, Dict, Any
from datetime import datetime, date, time, timedelta, timezone
import pytz
from enum import Enum
import calendar
import logging
import warnings

logger = logging.getLogger(__name__)

# Eastern Time Migration Constants
EASTERN_TIMEZONE = pytz.timezone('US/Eastern')
DEFAULT_TIMEZONE = EASTERN_TIMEZONE  # Changed from UTC
MIGRATION_DATE = datetime(2025, 8, 26, tzinfo=pytz.timezone('US/Eastern'))


class TimeZone(Enum):
    """Common timezone definitions - Eastern Time First"""
    EASTERN = "US/Eastern"  # Primary timezone
    UTC = "UTC"
    CST = "US/Central"
    MST = "US/Mountain"
    PST = "US/Pacific"
    LONDON = "Europe/London"
    TOKYO = "Asia/Tokyo"
    HONG_KONG = "Asia/Hong_Kong"
    SYDNEY = "Australia/Sydney"


class TimezoneConflictResolver:
    """Resolver for mixed timezone scenarios with Eastern Time precedence"""
    
    def __init__(self, default_tz: str = "US/Eastern"):
        self.default_tz = pytz.timezone(default_tz)
        self.conflict_log = []
    
    def resolve_mixed_timezones(self, timestamps: List[datetime]) -> List[datetime]:
        """Convert all timestamps to Eastern Time with conflict tracking"""
        resolved = []
        conflicts = []
        
        for i, ts in enumerate(timestamps):
            if ts.tzinfo is None:
                # Naive datetime - assume Eastern
                resolved_ts = self.default_tz.localize(ts)
                self.conflict_log.append({
                    'index': i,
                    'original': ts,
                    'resolved': resolved_ts,
                    'action': 'localized_as_eastern',
                    'confidence': 'assumed'
                })
                resolved.append(resolved_ts)
            elif ts.tzinfo != self.default_tz:
                # Different timezone - convert to Eastern
                resolved_ts = ts.astimezone(self.default_tz)
                self.conflict_log.append({
                    'index': i,
                    'original': ts,
                    'resolved': resolved_ts,
                    'action': 'converted_to_eastern',
                    'confidence': 'high'
                })
                resolved.append(resolved_ts)
            else:
                # Already Eastern
                resolved.append(ts)
        
        return resolved
    
    def get_conflict_summary(self) -> Dict[str, Any]:
        """Get summary of timezone conflicts encountered"""
        return {
            'total_conflicts': len(self.conflict_log),
            'conflicts_by_type': {
                'naive_datetime': len([c for c in self.conflict_log if c['action'] == 'localized_as_eastern']),
                'timezone_conversion': len([c for c in self.conflict_log if c['action'] == 'converted_to_eastern'])
            },
            'conflicts': self.conflict_log
        }


class MarketHours:
    """Market hours configuration - Eastern Time Standard"""
    
    def __init__(self, 
                 pre_market_start: time = time(4, 0, 0),
                 market_open: time = time(9, 30, 0),
                 market_close: time = time(16, 0, 0),
                 after_hours_end: time = time(20, 0, 0),
                 timezone: str = "US/Eastern"):
        """
        Initialize market hours in Eastern Time
        
        Args:
            pre_market_start: Pre-market start time (4:00 AM ET)
            market_open: Regular market open time (9:30 AM ET)
            market_close: Regular market close time (4:00 PM ET)
            after_hours_end: After-hours end time (8:00 PM ET)
            timezone: Market timezone (MUST be US/Eastern)
        """
        if timezone != "US/Eastern":
            logger.warning(f"Market timezone changed from {timezone} to US/Eastern for consistency")
        
        self.pre_market_start = pre_market_start
        self.market_open = market_open
        self.market_close = market_close
        self.after_hours_end = after_hours_end
        self.timezone = EASTERN_TIMEZONE  # Force Eastern Time
    
    def is_regular_hours(self, dt: datetime) -> bool:
        """Check if datetime is during regular market hours (9:30 AM - 4:00 PM ET)"""
        market_dt = self.to_eastern_time(dt)
        return self.market_open <= market_dt.time() <= self.market_close
    
    def is_pre_market(self, dt: datetime) -> bool:
        """Check if datetime is during pre-market hours (4:00 AM - 9:30 AM ET)"""
        market_dt = self.to_eastern_time(dt)
        return self.pre_market_start <= market_dt.time() < self.market_open
    
    def is_after_hours(self, dt: datetime) -> bool:
        """Check if datetime is during after-hours (4:00 PM - 8:00 PM ET)"""
        market_dt = self.to_eastern_time(dt)
        return self.market_close < market_dt.time() <= self.after_hours_end
    
    def is_extended_hours(self, dt: datetime) -> bool:
        """Check if datetime is during extended hours (pre + after)"""
        return self.is_pre_market(dt) or self.is_after_hours(dt)
    
    def is_market_open(self, dt: datetime, include_extended: bool = False) -> bool:
        """Check if market is open at given datetime"""
        if include_extended:
            return (self.is_regular_hours(dt) or 
                   self.is_pre_market(dt) or 
                   self.is_after_hours(dt))
        return self.is_regular_hours(dt)
    
    def to_eastern_time(self, dt: datetime) -> datetime:
        """Convert datetime to Eastern Time (replaces to_market_time)"""
        if dt.tzinfo is None:
            # Assume Eastern Time for naive datetime
            logger.debug(f"Localizing naive datetime {dt} as Eastern Time")
            return EASTERN_TIMEZONE.localize(dt)
        return dt.astimezone(EASTERN_TIMEZONE)
    
    def to_market_time(self, dt: datetime) -> datetime:
        """Convert datetime to market timezone (deprecated - use to_eastern_time)"""
        warnings.warn("to_market_time is deprecated, use to_eastern_time instead", 
                     DeprecationWarning, stacklevel=2)
        return self.to_eastern_time(dt)
    
    def get_next_market_open(self, from_dt: Optional[datetime] = None) -> datetime:
        """Get next market open datetime in Eastern Time"""
        if from_dt is None:
            from_dt = now_eastern()
        else:
            from_dt = self.to_eastern_time(from_dt)
        
        # Start from current date
        current_date = from_dt.date()
        
        # Check if market is already open today
        market_open_today = EASTERN_TIMEZONE.localize(
            datetime.combine(current_date, self.market_open)
        )
        
        if from_dt < market_open_today and not is_market_holiday(current_date):
            return market_open_today
        
        # Find next business day
        next_date = current_date + timedelta(days=1)
        while is_weekend(next_date) or is_market_holiday(next_date):
            next_date += timedelta(days=1)
        
        return EASTERN_TIMEZONE.localize(
            datetime.combine(next_date, self.market_open)
        )
    
    def get_next_market_close(self, from_dt: Optional[datetime] = None) -> datetime:
        """Get next market close datetime in Eastern Time"""
        if from_dt is None:
            from_dt = now_eastern()
        else:
            from_dt = self.to_eastern_time(from_dt)
        
        current_date = from_dt.date()
        
        # Check if market closes today
        market_close_today = EASTERN_TIMEZONE.localize(
            datetime.combine(current_date, self.market_close)
        )
        
        if (from_dt < market_close_today and 
            not is_weekend(current_date) and 
            not is_market_holiday(current_date)):
            return market_close_today
        
        # Find next business day close
        next_date = current_date + timedelta(days=1)
        while is_weekend(next_date) or is_market_holiday(next_date):
            next_date += timedelta(days=1)
        
        return EASTERN_TIMEZONE.localize(
            datetime.combine(next_date, self.market_close)
        )
    
    def is_open(self, dt: Optional[datetime] = None) -> bool:
        """Check if market is currently open (test compatibility)"""
        if dt is None:
            dt = now_eastern()
        return self.is_market_open(dt)
    
    def is_trading_day(self, test_date: date) -> bool:
        """Check if given date is a trading day (test compatibility)"""
        return not is_weekend(test_date) and not is_market_holiday(test_date)


# Market holidays for 2025 (updated for migration year)
MARKET_HOLIDAYS_2025 = [
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # Martin Luther King Jr. Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
]

MARKET_HOLIDAYS_2024 = [
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # Martin Luther King Jr. Day
    date(2024, 2, 19),  # Presidents' Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 11, 28), # Thanksgiving
    date(2024, 12, 25), # Christmas
]

# Default to current year holidays
CURRENT_YEAR_HOLIDAYS = MARKET_HOLIDAYS_2025


# Core Eastern Time Functions
def now_eastern() -> datetime:
    """Return current time in Eastern timezone"""
    return datetime.now(EASTERN_TIMEZONE)


def now_utc() -> datetime:
    """Return timezone-aware current UTC datetime (legacy support)"""
    import warnings
    warnings.warn("now_utc() is deprecated, use now_et() for Eastern Time", 
                 DeprecationWarning, stacklevel=2)
    return datetime.now(timezone.utc)


def eastern_to_utc(eastern_dt: datetime) -> datetime:
    """Convert Eastern Time datetime to UTC"""
    if eastern_dt.tzinfo is None:
        eastern_dt = EASTERN_TIMEZONE.localize(eastern_dt)
    return eastern_dt.astimezone(pytz.UTC)


def utc_to_eastern(utc_dt: datetime) -> datetime:
    """Convert UTC datetime to Eastern Time"""
    if utc_dt.tzinfo is None:
        utc_dt = pytz.UTC.localize(utc_dt)
    return utc_dt.astimezone(EASTERN_TIMEZONE)


def is_weekend(check_date: date) -> bool:
    """Check if date is a weekend"""
    return check_date.weekday() >= 5  # Saturday=5, Sunday=6


def is_market_holiday(check_date: date, holidays: Optional[List[date]] = None) -> bool:
    """Check if date is a market holiday"""
    if holidays is None:
        holidays = CURRENT_YEAR_HOLIDAYS
    return check_date in holidays


def is_business_day(check_date: date, holidays: Optional[List[date]] = None) -> bool:
    """Check if date is a business day"""
    return not is_weekend(check_date) and not is_market_holiday(check_date, holidays)


def get_business_days(start_date: date, end_date: date, 
                     holidays: Optional[List[date]] = None) -> List[date]:
    """Get list of business days between start and end dates"""
    business_days = []
    current_date = start_date
    
    while current_date <= end_date:
        if is_business_day(current_date, holidays):
            business_days.append(current_date)
        current_date += timedelta(days=1)
    
    return business_days


def get_next_business_day(from_date: date, holidays: Optional[List[date]] = None) -> date:
    """Get next business day from given date"""
    next_date = from_date + timedelta(days=1)
    while not is_business_day(next_date, holidays):
        next_date += timedelta(days=1)
    return next_date


def get_previous_business_day(from_date: date, holidays: Optional[List[date]] = None) -> date:
    """Get previous business day from given date"""
    prev_date = from_date - timedelta(days=1)
    while not is_business_day(prev_date, holidays):
        prev_date -= timedelta(days=1)
    return prev_date


def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
    """
    Convert datetime from one timezone to another
    
    Args:
        dt: Datetime to convert
        from_tz: Source timezone string
        to_tz: Target timezone string
        
    Returns:
        Converted datetime
    """
    from_timezone = pytz.timezone(from_tz)
    to_timezone = pytz.timezone(to_tz)
    
    # Localize if naive
    if dt.tzinfo is None:
        dt = from_timezone.localize(dt)
    
    return dt.astimezone(to_timezone)


def utc_to_market_time(utc_dt: datetime, market_tz: str = "US/Eastern") -> datetime:
    """Convert UTC datetime to market timezone (now defaults to Eastern)"""
    if utc_dt.tzinfo is None:
        utc_dt = pytz.UTC.localize(utc_dt)
    
    market_timezone = pytz.timezone(market_tz)
    return utc_dt.astimezone(market_timezone)


def market_time_to_utc(market_dt: datetime, market_tz: str = "US/Eastern") -> datetime:
    """Convert market timezone datetime to UTC (now defaults to Eastern)"""
    if market_dt.tzinfo is None:
        market_timezone = pytz.timezone(market_tz)
        market_dt = market_timezone.localize(market_dt)
    
    return market_dt.astimezone(pytz.UTC)


def parse_timestamp(timestamp: Union[str, datetime, int, float], 
                   default_tz: Optional[str] = "US/Eastern") -> datetime:
    """
    Parse various timestamp formats into datetime with Eastern Time default
    
    Args:
        timestamp: Timestamp to parse
        default_tz: Default timezone if none specified (now defaults to Eastern)
        
    Returns:
        Parsed datetime in Eastern timezone
    """
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None and default_tz:
            # Localize naive datetime to Eastern
            tz = pytz.timezone(default_tz)
            return tz.localize(timestamp)
        return timestamp
    
    if isinstance(timestamp, (int, float)):
        # Unix timestamp - convert to Eastern
        utc_dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        return utc_dt.astimezone(EASTERN_TIMEZONE)
    
    if isinstance(timestamp, str):
        # Try various string formats
        try:
            # ISO format
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.astimezone(EASTERN_TIMEZONE) if dt.tzinfo else EASTERN_TIMEZONE.localize(dt)
        except ValueError:
            pass
        
        try:
            # Common formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%d',
                '%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    if default_tz:
                        tz = pytz.timezone(default_tz)
                        dt = tz.localize(dt)
                    return dt
                except ValueError:
                    continue
        except Exception:
            pass
    
    raise ValueError(f"Unable to parse timestamp: {timestamp}")


def format_timestamp(dt: datetime, format_type: str = 'iso') -> str:
    """
    Format datetime into string with Eastern Time awareness
    
    Args:
        dt: Datetime to format
        format_type: Format type ('iso', 'eastern', 'market', 'date', 'time')
        
    Returns:
        Formatted timestamp string
    """
    # Ensure we have Eastern timezone
    if dt.tzinfo is None:
        dt = EASTERN_TIMEZONE.localize(dt)
    elif dt.tzinfo != EASTERN_TIMEZONE:
        dt = dt.astimezone(EASTERN_TIMEZONE)
    
    if format_type == 'iso':
        return dt.isoformat()
    elif format_type == 'eastern' or format_type == 'market':
        return dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    elif format_type == 'date':
        return dt.strftime('%Y-%m-%d')
    elif format_type == 'time':
        return dt.strftime('%H:%M:%S')
    else:
        return str(dt)


def get_trading_session(dt: datetime, market_hours: Optional[MarketHours] = None) -> str:
    """
    Get trading session for datetime in Eastern Time
    
    Args:
        dt: Datetime to check (converted to Eastern)
        market_hours: Market hours configuration
        
    Returns:
        Session name ('pre_market', 'regular', 'after_hours', 'closed')
    """
    if market_hours is None:
        market_hours = MarketHours()
    
    # Ensure Eastern Time
    eastern_dt = market_hours.to_eastern_time(dt)
    
    if market_hours.is_regular_hours(eastern_dt):
        return 'regular'
    elif market_hours.is_pre_market(eastern_dt):
        return 'pre_market'
    elif market_hours.is_after_hours(eastern_dt):
        return 'after_hours'
    else:
        return 'closed'


def get_time_range_boundaries(start: datetime, end: datetime, 
                            interval: str = 'day') -> List[datetime]:
    """
    Get time boundaries for a range at specified interval
    
    Args:
        start: Start datetime
        end: End datetime
        interval: Interval type ('minute', 'hour', 'day', 'week', 'month')
        
    Returns:
        List of boundary datetimes
    """
    boundaries = []
    current = start
    
    if interval == 'minute':
        delta = timedelta(minutes=1)
    elif interval == 'hour':
        delta = timedelta(hours=1)
    elif interval == 'day':
        delta = timedelta(days=1)
    elif interval == 'week':
        delta = timedelta(weeks=1)
    elif interval == 'month':
        # Handle months separately
        while current <= end:
            boundaries.append(current)
            # Add one month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return boundaries
    else:
        raise ValueError(f"Unsupported interval: {interval}")
    
    while current <= end:
        boundaries.append(current)
        current += delta
    
    return boundaries


def calculate_time_metrics(timestamps: List[datetime]) -> dict:
    """
    Calculate time-related metrics from timestamps
    
    Args:
        timestamps: List of timestamps
        
    Returns:
        Dictionary of metrics
    """
    if not timestamps:
        return {}
    
    sorted_timestamps = sorted(timestamps)
    
    metrics = {
        'count': len(timestamps),
        'start_time': sorted_timestamps[0],
        'end_time': sorted_timestamps[-1],
        'duration_seconds': (sorted_timestamps[-1] - sorted_timestamps[0]).total_seconds(),
    }
    
    # Calculate intervals between timestamps
    if len(sorted_timestamps) > 1:
        intervals = []
        for i in range(1, len(sorted_timestamps)):
            interval = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        metrics.update({
            'average_interval_seconds': sum(intervals) / len(intervals),
            'min_interval_seconds': min(intervals),
            'max_interval_seconds': max(intervals),
            'frequency_per_minute': len(timestamps) / (metrics['duration_seconds'] / 60) if metrics['duration_seconds'] > 0 else 0
        })
    
    return metrics


def get_market_calendar(year: int, holidays: Optional[List[date]] = None) -> List[date]:
    """
    Get market calendar (business days) for a year in Eastern timezone context
    
    Args:
        year: Year to get calendar for
        holidays: List of holidays to exclude (defaults to current year)
        
    Returns:
        List of business days
    """
    if holidays is None:
        holidays = CURRENT_YEAR_HOLIDAYS
    
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    
    return get_business_days(start_date, end_date, holidays)

# --- Lightweight UTC helpers (referenced by provider hardening patch) ---
def to_rfc3339(dt: datetime, milliseconds: bool = False) -> str:
    """Serialize datetime to RFC3339 with trailing Z.

    Args:
        dt: datetime (aware or naive; naive assumed Eastern then converted to UTC)
        milliseconds: include millisecond precision
    """
    if dt.tzinfo is None:
        # Assume Eastern Time, then convert to UTC
        dt = EASTERN_TIMEZONE.localize(dt)
    dt = dt.astimezone(timezone.utc)
    if milliseconds:
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    return dt.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'


def parse_iso8601_utc(value) -> Optional[datetime]:
    """Parse ISO8601 string to Eastern Time datetime.
    
    Args:
        value: ISO8601 string, datetime object, or other value
        
    Returns:
        Eastern Time datetime or None if parsing fails
    """
    if not value:
        return None
    if isinstance(value, datetime):
        if value.tzinfo:
            return value.astimezone(EASTERN_TIMEZONE)
        else:
            return EASTERN_TIMEZONE.localize(value)
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if dt.tzinfo:
            return dt.astimezone(EASTERN_TIMEZONE)
        else:
            return EASTERN_TIMEZONE.localize(dt)
    except Exception:
        return None


class TimeRangeIterator:
    """Iterator for time ranges"""
    
    def __init__(self, start: datetime, end: datetime, step: timedelta):
        """
        Initialize time range iterator
        
        Args:
            start: Start datetime
            end: End datetime
            step: Time step
        """
        self.start = start
        self.end = end
        self.step = step
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        
        result = self.current
        self.current += self.step
        return result


# Default market hours instance - Eastern Time
DEFAULT_MARKET_HOURS = MarketHours(timezone="US/Eastern")


# Compatibility aliases for tests
now_et = now_eastern  # Alias for test compatibility
to_et_from_utc = utc_to_eastern  # Alias for test compatibility
to_utc_from_et = eastern_to_utc  # Alias for test compatibility
format_et_timestamp = format_timestamp  # Alias for test compatibility
parse_iso8601_et = parse_iso8601_utc  # Alias for test compatibility
MarketHoursET = MarketHours  # Alias for test compatibility

def get_market_open_et(test_date: date) -> datetime:
    """Get market open time for a specific date in Eastern Time (test compatibility)"""
    market_hours = MarketHours()
    market_open_dt = EASTERN_TIMEZONE.localize(
        datetime.combine(test_date, market_hours.market_open)
    )
    return market_open_dt

def get_market_close_et(test_date: date) -> datetime:
    """Get market close time for a specific date in Eastern Time (test compatibility)"""
    market_hours = MarketHours()
    market_close_dt = EASTERN_TIMEZONE.localize(
        datetime.combine(test_date, market_hours.market_close)
    )
    return market_close_dt


# Example Eastern Time usage
def example_eastern_time_usage():
    """Example usage of Eastern Time utilities"""
    
    # Current Eastern Time
    now_et = now_eastern()
    print(f"Current Eastern Time: {now_et}")
    
    # Market hours checking
    market_hours = MarketHours()
    print(f"Is regular hours: {market_hours.is_regular_hours(now_et)}")
    print(f"Trading session: {get_trading_session(now_et)}")
    
    # Next market events
    next_open = market_hours.get_next_market_open()
    next_close = market_hours.get_next_market_close()
    print(f"Next market open (ET): {next_open}")
    print(f"Next market close (ET): {next_close}")
    
    # Business days
    today = date.today()
    next_bday = get_next_business_day(today)
    print(f"Next business day: {next_bday}")
    
    # Timezone conversion
    utc_time = datetime.now(pytz.UTC)
    eastern_time = utc_to_eastern(utc_time)
    print(f"UTC: {utc_time}, Eastern: {eastern_time}")
    
    # Time range in Eastern
    start = EASTERN_TIMEZONE.localize(datetime(2025, 8, 26, 9, 30))
    end = EASTERN_TIMEZONE.localize(datetime(2025, 8, 26, 16, 0))
    
    for dt in TimeRangeIterator(start, end, timedelta(hours=1)):
        print(f"Hour boundary (ET): {dt}")


# Additional test compatibility functions
def is_market_hours_et():
    """Check if current Eastern Time is during market hours."""
    current_et = now_eastern()
    market_hours = MarketHours()
    return market_hours.is_market_open(current_et)

def format_et_timestamp_enhanced(dt, format_str="%Y-%m-%d %H:%M:%S", include_tz=False):
    """Enhanced format_et_timestamp with include_tz parameter for test compatibility."""
    if dt.tzinfo is None:
        dt = EASTERN_TIMEZONE.localize(dt)
    elif dt.tzinfo != EASTERN_TIMEZONE:
        dt = dt.astimezone(EASTERN_TIMEZONE)
    
    formatted = dt.strftime(format_str)
    if include_tz:
        tz_suffix = "EST" if not dt.dst() else "EDT" 
        formatted += f" {tz_suffix}"
    return formatted

# Override the compatibility alias to use enhanced version
format_et_timestamp = format_et_timestamp_enhanced

if __name__ == "__main__":
    example_eastern_time_usage()
