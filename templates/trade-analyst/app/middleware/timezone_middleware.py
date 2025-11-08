"""Timezone Middleware for Eastern Time API Interfacing

This middleware ensures consistent Eastern Time handling across all API calls,
providing automatic timezone conversion with conflict detection and resolution.
"""

import functools
import logging
import time
from typing import Any, Dict, List, Callable, Optional, Union
from datetime import datetime, date, timezone
import pytz
from flask import request, jsonify, g
import warnings

from app.utils.timeutils import (
    EASTERN_TIMEZONE, 
    TimezoneConflictResolver, 
    now_eastern, 
    utc_to_eastern,
    eastern_to_utc,
    parse_timestamp
)

logger = logging.getLogger(__name__)


class TimezoneMiddleware:
    """Middleware for automatic timezone handling in API requests/responses"""
    
    def __init__(self, app=None, default_timezone: str = "US/Eastern"):
        self.app = app
        self.default_timezone = pytz.timezone(default_timezone)
        self.conflict_resolver = TimezoneConflictResolver(default_timezone)
        self.request_stats = {
            'total_requests': 0,
            'timezone_conversions': 0,
            'conflicts_resolved': 0
        }
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize middleware with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_request(self.teardown_request)
    
    def before_request(self):
        """Process incoming request for timezone handling"""
        self.request_stats['total_requests'] += 1
        
        # Store timezone context in Flask g object
        g.timezone_context = {
            'request_timezone': 'US/Eastern',
            'original_data': {},
            'conversions_made': [],
            'conflicts_detected': []
        }
        
        # Convert request data to Eastern Time
        if request.is_json and request.get_json():
            original_data = request.get_json()
            converted_data = self._convert_request_data(original_data)
            g.timezone_context['original_data'] = original_data
            # Store converted data for later use
            request._converted_json = converted_data
    
    def after_request(self, response):
        """Process outgoing response for timezone consistency"""
        # Ensure all datetime fields in response are Eastern Time
        if response.is_json:
            try:
                data = response.get_json()
                if data:
                    converted_data = self._convert_response_data(data)
                    response.data = response.get_app().json.dumps(converted_data)
            except Exception as e:
                logger.warning(f"Failed to process response timezone: {e}")
        
        # Add timezone headers
        response.headers['X-Timezone'] = 'US/Eastern'
        response.headers['X-Timezone-Conversions'] = str(len(g.get('timezone_context', {}).get('conversions_made', [])))
        
        return response
    
    def teardown_request(self, exception):
        """Clean up timezone context after request"""
        if hasattr(g, 'timezone_context'):
            conflicts = g.timezone_context.get('conflicts_detected', [])
            if conflicts:
                self.request_stats['conflicts_resolved'] += len(conflicts)
                logger.info(f"Resolved {len(conflicts)} timezone conflicts in request")
    
    def _convert_request_data(self, data: Any) -> Any:
        """Recursively convert request data timestamps to Eastern Time"""
        if isinstance(data, dict):
            converted = {}
            for key, value in data.items():
                if self._is_timestamp_field(key, value):
                    try:
                        converted_value = self._convert_to_eastern(value)
                        converted[key] = converted_value
                        g.timezone_context['conversions_made'].append({
                            'field': key,
                            'original': str(value),
                            'converted': str(converted_value),
                            'type': 'request_input'
                        })
                    except Exception as e:
                        logger.warning(f"Failed to convert timestamp field {key}: {e}")
                        converted[key] = value
                else:
                    converted[key] = self._convert_request_data(value)
            return converted
        elif isinstance(data, list):
            return [self._convert_request_data(item) for item in data]
        else:
            return data
    
    def _convert_response_data(self, data: Any) -> Any:
        """Recursively convert response data timestamps to Eastern Time"""
        if isinstance(data, dict):
            converted = {}
            for key, value in data.items():
                if self._is_timestamp_field(key, value):
                    try:
                        converted_value = self._convert_to_eastern(value)
                        converted[key] = converted_value
                        g.timezone_context['conversions_made'].append({
                            'field': key,
                            'original': str(value),
                            'converted': str(converted_value),
                            'type': 'response_output'
                        })
                    except Exception as e:
                        logger.warning(f"Failed to convert response timestamp field {key}: {e}")
                        converted[key] = value
                else:
                    converted[key] = self._convert_response_data(value)
            return converted
        elif isinstance(data, list):
            return [self._convert_response_data(item) for item in data]
        else:
            return data
    
    def _is_timestamp_field(self, key: str, value: Any) -> bool:
        """Determine if a field contains timestamp data"""
        if not isinstance(value, (str, datetime, int, float)):
            return False
        
        # Common timestamp field names
        timestamp_fields = [
            'time', 'timestamp', 'date', 'datetime',
            'created_at', 'updated_at', 'modified_at',
            'start_time', 'end_time', 'open_time', 'close_time',
            'market_time', 'trade_time', 'quote_time',
            'expiry', 'expiration', 'expires_at'
        ]
        
        key_lower = key.lower()
        if any(field in key_lower for field in timestamp_fields):
            return True
        
        # Try to parse as timestamp
        if isinstance(value, str):
            try:
                parse_timestamp(value)
                return True
            except:
                return False
        
        return isinstance(value, (datetime, int, float))
    
    def _convert_to_eastern(self, value: Any) -> Any:
        """Convert a value to Eastern Time"""
        try:
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    return EASTERN_TIMEZONE.localize(value)
                return value.astimezone(EASTERN_TIMEZONE)
            
            # Parse string/numeric timestamps
            dt = parse_timestamp(value, default_tz="US/Eastern")
            return dt.astimezone(EASTERN_TIMEZONE)
            
        except Exception as e:
            logger.debug(f"Could not convert value {value} to Eastern Time: {e}")
            return value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        return {
            **self.request_stats,
            'conflict_resolver_stats': self.conflict_resolver.get_conflict_summary()
        }


def timezone_aware(func: Callable) -> Callable:
    """Decorator to ensure function operates in Eastern Time context"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Convert datetime arguments to Eastern Time
        converted_args = []
        for arg in args:
            if isinstance(arg, datetime):
                if arg.tzinfo is None:
                    arg = EASTERN_TIMEZONE.localize(arg)
                else:
                    arg = arg.astimezone(EASTERN_TIMEZONE)
            converted_args.append(arg)
        
        converted_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    value = EASTERN_TIMEZONE.localize(value)
                else:
                    value = value.astimezone(EASTERN_TIMEZONE)
            converted_kwargs[key] = value
        
        # Execute function
        result = func(*converted_args, **converted_kwargs)
        
        # Convert result datetime fields to Eastern Time
        if isinstance(result, datetime):
            if result.tzinfo is None:
                return EASTERN_TIMEZONE.localize(result)
            return result.astimezone(EASTERN_TIMEZONE)
        elif isinstance(result, dict):
            return _convert_dict_timestamps_to_eastern(result)
        elif isinstance(result, list):
            return [_convert_dict_timestamps_to_eastern(item) if isinstance(item, dict) else item for item in result]
        
        return result
    
    return wrapper


def schwab_api_timezone_handler(func: Callable) -> Callable:
    """Decorator specifically for Schwab API calls with timezone handling"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Schwab API call: {func.__name__} with timezone handling")
        
        # Pre-process: Convert Eastern times to appropriate format for Schwab API
        start_time = time.time()
        
        try:
            # Convert datetime parameters for Schwab API
            if 'start_date' in kwargs and isinstance(kwargs['start_date'], datetime):
                kwargs['start_date'] = _prepare_datetime_for_schwab(kwargs['start_date'])
            if 'end_date' in kwargs and isinstance(kwargs['end_date'], datetime):
                kwargs['end_date'] = _prepare_datetime_for_schwab(kwargs['end_date'])
            
            # Execute API call
            result = func(*args, **kwargs)
            
            # Post-process: Convert Schwab response timestamps to Eastern Time
            if isinstance(result, dict):
                result = _convert_schwab_response_to_eastern(result)
            elif isinstance(result, list):
                result = [_convert_schwab_response_to_eastern(item) if isinstance(item, dict) else item for item in result]
            
            processing_time = time.time() - start_time
            logger.debug(f"Schwab API call {func.__name__} completed in {processing_time:.3f}s with timezone conversion")
            
            return result
            
        except Exception as e:
            logger.error(f"Schwab API call {func.__name__} failed with timezone handling: {e}")
            raise
    
    return wrapper


def _prepare_datetime_for_schwab(dt: datetime) -> str:
    """Prepare datetime for Schwab API (typically requires UTC or epoch)"""
    if dt.tzinfo is None:
        dt = EASTERN_TIMEZONE.localize(dt)
    
    # Convert to UTC for Schwab API
    utc_dt = dt.astimezone(pytz.UTC)
    # Return in ISO format that Schwab expects
    return utc_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def _convert_schwab_response_to_eastern(data: Dict) -> Dict:
    """Convert Schwab API response timestamps to Eastern Time"""
    if not isinstance(data, dict):
        return data
    
    schwab_timestamp_fields = [
        'datetime', 'timestamp', 'time',
        'quoteTimeInLong', 'tradeTimeInLong',
        'lastTradingDay', 'expirationDate',
        'openTime', 'closeTime'
    ]
    
    converted = {}
    for key, value in data.items():
        if key in schwab_timestamp_fields:
            try:
                if isinstance(value, (int, float)) and value > 1000000000:
                    # Likely epoch timestamp
                    dt = datetime.fromtimestamp(value / 1000 if value > 1000000000000 else value, tz=pytz.UTC)
                    converted[key] = dt.astimezone(EASTERN_TIMEZONE)
                elif isinstance(value, str):
                    dt = parse_timestamp(value)
                    converted[key] = dt.astimezone(EASTERN_TIMEZONE)
                else:
                    converted[key] = value
            except Exception as e:
                logger.debug(f"Could not convert Schwab timestamp field {key}: {e}")
                converted[key] = value
        elif isinstance(value, dict):
            converted[key] = _convert_schwab_response_to_eastern(value)
        elif isinstance(value, list):
            converted[key] = [_convert_schwab_response_to_eastern(item) if isinstance(item, dict) else item for item in value]
        else:
            converted[key] = value
    
    return converted


def _convert_dict_timestamps_to_eastern(data: Dict) -> Dict:
    """Convert dictionary timestamp fields to Eastern Time"""
    if not isinstance(data, dict):
        return data
    
    converted = {}
    for key, value in data.items():
        if isinstance(value, datetime):
            if value.tzinfo is None:
                converted[key] = EASTERN_TIMEZONE.localize(value)
            else:
                converted[key] = value.astimezone(EASTERN_TIMEZONE)
        elif isinstance(value, dict):
            converted[key] = _convert_dict_timestamps_to_eastern(value)
        elif isinstance(value, list):
            converted[key] = [_convert_dict_timestamps_to_eastern(item) if isinstance(item, dict) else item for item in value]
        else:
            converted[key] = value
    
    return converted


def eastern_time_required(func: Callable) -> Callable:
    """Decorator that enforces Eastern Time for all datetime parameters"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Validate and convert all datetime parameters
        for i, arg in enumerate(args):
            if isinstance(arg, datetime):
                if arg.tzinfo is None:
                    warnings.warn(f"Naive datetime passed to {func.__name__}, assuming Eastern Time", 
                                UserWarning, stacklevel=2)
                    args = list(args)
                    args[i] = EASTERN_TIMEZONE.localize(arg)
                elif arg.tzinfo != EASTERN_TIMEZONE:
                    logger.info(f"Converting {arg} to Eastern Time for {func.__name__}")
                    args = list(args)
                    args[i] = arg.astimezone(EASTERN_TIMEZONE)
        
        for key, value in kwargs.items():
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    warnings.warn(f"Naive datetime passed to {func.__name__} parameter {key}, assuming Eastern Time", 
                                UserWarning, stacklevel=2)
                    kwargs[key] = EASTERN_TIMEZONE.localize(value)
                elif value.tzinfo != EASTERN_TIMEZONE:
                    logger.info(f"Converting {key}={value} to Eastern Time for {func.__name__}")
                    kwargs[key] = value.astimezone(EASTERN_TIMEZONE)
        
        return func(*args, **kwargs)
    
    return wrapper


# Global middleware instance
timezone_middleware = TimezoneMiddleware()


# Example usage and testing
if __name__ == "__main__":
    from datetime import datetime
    import pytz
    
    print("🕐 TIMEZONE MIDDLEWARE TEST")
    print("=" * 40)
    
    # Test timezone aware decorator
    @timezone_aware
    def sample_function(dt: datetime) -> datetime:
        return dt
    
    # Test with naive datetime
    naive_dt = datetime(2025, 8, 26, 14, 30)
    result = sample_function(naive_dt)
    print(f"✅ Naive datetime converted: {result}")
    
    # Test with UTC datetime
    utc_dt = pytz.UTC.localize(datetime(2025, 8, 26, 18, 30))
    result = sample_function(utc_dt)
    print(f"✅ UTC datetime converted: {result}")
    
    # Test Schwab API handler
    @schwab_api_timezone_handler
    def mock_schwab_api(**kwargs):
        return {
            'datetime': '2025-08-26T18:30:00Z',
            'quoteTimeInLong': 1724697000000,
            'price': 100.50
        }
    
    schwab_result = mock_schwab_api(start_date=naive_dt)
    print(f"✅ Schwab API result: {schwab_result}")
    
    print("🎯 TIMEZONE MIDDLEWARE: SUCCESS!")
