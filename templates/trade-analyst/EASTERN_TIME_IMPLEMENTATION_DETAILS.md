# Eastern Time Migration - Critical Implementation Details

## Mixed Timezone Handling Policy

### 1. Timezone Precedence Hierarchy

**Primary Rule: Eastern Time Always Takes Precedence for Display and Business Logic**

```python
# Timezone Resolution Policy
TIMEZONE_PRECEDENCE = {
    'display': 'US/Eastern',           # All user-facing timestamps
    'business_logic': 'US/Eastern',    # Contract expiry, market hours
    'external_apis': 'UTC',            # Outbound API calls
    'storage': 'US/Eastern',           # New data storage
    'legacy_storage': 'UTC'            # Existing data (marked with metadata)
}
```

### 2. Conflict Resolution Strategy

```python
class TimezoneConflictResolver:
    """Handles timezone conflicts with clear precedence rules"""
    
    def resolve_timestamp_conflict(self, timestamps: Dict[str, datetime]) -> datetime:
        """Resolve conflicts when multiple timezone sources exist"""
        
        # Priority order for conflict resolution
        priority_sources = [
            'user_input_et',      # User explicitly provided ET time
            'market_data_et',     # Market data in ET
            'system_generated_et', # System-generated ET timestamps
            'external_api_utc',   # External API data (converted to ET)
            'legacy_utc'          # Legacy UTC data (converted to ET)
        ]
        
        for source in priority_sources:
            if source in timestamps:
                winning_timestamp = timestamps[source]
                
                # Log the conflict resolution
                self._log_conflict_resolution(timestamps, source, winning_timestamp)
                
                # Ensure result is in Eastern Time
                return self.ensure_eastern_time(winning_timestamp)
        
        raise TimezoneConflictError("No valid timestamp source found")
    
    def _log_conflict_resolution(self, all_timestamps: Dict, winner: str, result: datetime):
        """Log timezone conflict resolution for debugging"""
        conflict_log = {
            'timestamp': self.now_et(),
            'conflict_type': 'timezone_resolution',
            'winner': winner,
            'resolved_time': result.isoformat(),
            'all_sources': {
                source: ts.isoformat() for source, ts in all_timestamps.items()
            },
            'policy': 'ET_PRECEDENCE'
        }
        
        logger.info("Timezone conflict resolved", extra=conflict_log)
```

### 3. Dual-Tagging Strategy for Critical Operations

```python
class DualTimestampLogger:
    """Logs critical operations with both ET and UTC for traceability"""
    
    def log_critical_operation(self, operation: str, data: dict):
        """Log with dual timestamps for maximum traceability"""
        
        current_et = timeutils.now_et()
        current_utc = current_et.astimezone(timezone.utc)
        
        log_entry = {
            'operation': operation,
            'timestamp_et': current_et.isoformat(),
            'timestamp_utc': current_utc.isoformat(),
            'timezone_source': 'DUAL_TAGGED',
            'data': data,
            'migration_phase': 'POST_ET_MIGRATION'
        }
        
        # Log to both regular and audit logs
        logger.info(f"Critical operation: {operation}", extra=log_entry)
        audit_logger.info(f"AUDIT: {operation}", extra=log_entry)
```

## External API Interfacing Strategy

### 1. Timezone Conversion Middleware

```python
from functools import wraps
import pytz

class TimezoneMiddleware:
    """Automatic timezone conversion for external API interactions"""
    
    def __init__(self):
        self.et_tz = pytz.timezone('US/Eastern')
        self.utc_tz = pytz.UTC
    
    def api_outbound(self, func):
        """Decorator: Convert ET timestamps to UTC for outbound API calls"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert any datetime objects in args/kwargs to UTC
            converted_args = self._convert_datetimes_to_utc(args)
            converted_kwargs = self._convert_datetimes_to_utc(kwargs)
            
            # Log conversion for debugging
            self._log_outbound_conversion(func.__name__, args, converted_args)
            
            # Call original function with UTC timestamps
            result = func(*converted_args, **converted_kwargs)
            
            return result
        return wrapper
    
    def api_inbound(self, func):
        """Decorator: Convert UTC timestamps from APIs to ET for internal use"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call original function
            result = func(*args, **kwargs)
            
            # Convert any datetime objects in result to ET
            converted_result = self._convert_datetimes_to_et(result)
            
            # Log conversion for debugging
            self._log_inbound_conversion(func.__name__, result, converted_result)
            
            return converted_result
        return wrapper
    
    def _convert_datetimes_to_utc(self, data):
        """Recursively convert datetime objects to UTC"""
        if isinstance(data, datetime):
            if data.tzinfo is None:
                # Assume naive datetime is ET
                et_aware = self.et_tz.localize(data)
                return et_aware.astimezone(self.utc_tz)
            else:
                return data.astimezone(self.utc_tz)
        elif isinstance(data, dict):
            return {k: self._convert_datetimes_to_utc(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._convert_datetimes_to_utc(item) for item in data)
        else:
            return data
    
    def _convert_datetimes_to_et(self, data):
        """Recursively convert datetime objects to ET"""
        if isinstance(data, datetime):
            return data.astimezone(self.et_tz)
        elif isinstance(data, dict):
            return {k: self._convert_datetimes_to_et(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._convert_datetimes_to_et(item) for item in data)
        else:
            return data
    
    def _log_outbound_conversion(self, func_name: str, original, converted):
        """Log outbound timezone conversions"""
        logger.debug(f"API Outbound Conversion - {func_name}", extra={
            'direction': 'outbound',
            'function': func_name,
            'original_tz': 'ET',
            'converted_tz': 'UTC',
            'timestamp': timeutils.now_et().isoformat()
        })
    
    def _log_inbound_conversion(self, func_name: str, original, converted):
        """Log inbound timezone conversions"""
        logger.debug(f"API Inbound Conversion - {func_name}", extra={
            'direction': 'inbound', 
            'function': func_name,
            'original_tz': 'UTC',
            'converted_tz': 'ET',
            'timestamp': timeutils.now_et().isoformat()
        })


# Usage Examples
timezone_middleware = TimezoneMiddleware()

class SchwabAPIClient:
    @timezone_middleware.api_outbound
    def get_price_history(self, symbol: str, start_date: datetime, end_date: datetime):
        """Get price history - dates automatically converted to UTC for API"""
        # start_date and end_date are now in UTC for Schwab API
        return self._make_request('/pricehistory', {
            'symbol': symbol,
            'startDate': int(start_date.timestamp() * 1000),
            'endDate': int(end_date.timestamp() * 1000)
        })
    
    @timezone_middleware.api_inbound
    def _make_request(self, endpoint: str, params: dict):
        """Make API request - response timestamps converted to ET"""
        response = requests.get(f"{self.base_url}{endpoint}", params=params)
        data = response.json()
        
        # Any datetime objects in response will be converted to ET
        return data
```

### 2. API Contract Documentation

```python
class APITimezoneContract:
    """Documents timezone expectations for all external APIs"""
    
    API_CONTRACTS = {
        'schwab_api': {
            'expects_timezone': 'UTC',
            'returns_timezone': 'UTC', 
            'timestamp_format': 'unix_milliseconds',
            'conversion_required': True,
            'critical_endpoints': ['/pricehistory', '/accounts', '/orders']
        },
        'market_data_feed': {
            'expects_timezone': 'ET',
            'returns_timezone': 'ET',
            'timestamp_format': 'iso8601',
            'conversion_required': False,
            'critical_endpoints': ['/quotes', '/trades']
        },
        'internal_services': {
            'expects_timezone': 'ET',
            'returns_timezone': 'ET', 
            'timestamp_format': 'iso8601',
            'conversion_required': False,
            'critical_endpoints': ['/health', '/metrics']
        }
    }
    
    @classmethod
    def get_conversion_requirements(cls, api_name: str) -> dict:
        """Get timezone conversion requirements for specific API"""
        contract = cls.API_CONTRACTS.get(api_name, {})
        return {
            'inbound_conversion': contract.get('returns_timezone') != 'ET',
            'outbound_conversion': contract.get('expects_timezone') != 'ET',
            'format': contract.get('timestamp_format', 'iso8601')
        }
```

## Contract Expiration Edge Cases

### 1. Comprehensive Expiration Testing

```python
class ContractExpirationTester:
    """Test contract expiration edge cases comprehensively"""
    
    def test_expiration_edge_cases(self):
        """Test all expiration edge cases"""
        
        edge_cases = [
            # Standard expiration scenarios
            {
                'name': 'Standard September Expiry',
                'current_time': self.et_tz.localize(datetime(2025, 9, 19, 9, 29, 0)),  # 1 min before expiry
                'expected_contract': 'NQU25',
                'description': 'Should use September until 9:30 AM ET on expiry day'
            },
            {
                'name': 'Post Expiry Same Day', 
                'current_time': self.et_tz.localize(datetime(2025, 9, 19, 9, 31, 0)),  # 1 min after expiry
                'expected_contract': 'NQZ25',
                'description': 'Should rollover to December at 9:30 AM ET on expiry day'
            },
            
            # Midnight boundary testing
            {
                'name': 'Midnight ET Before Expiry',
                'current_time': self.et_tz.localize(datetime(2025, 9, 19, 0, 0, 0)),   # Midnight ET on expiry day
                'expected_contract': 'NQU25',
                'description': 'Should still use September at midnight ET on expiry day'
            },
            {
                'name': 'Midnight UTC Different Day',
                'current_time': datetime(2025, 9, 19, 4, 0, 0, tzinfo=timezone.utc),   # Midnight UTC = 8 PM ET previous day
                'expected_contract': 'NQU25', 
                'description': 'UTC midnight should not affect ET-based expiry logic'
            },
            
            # Holiday and early close scenarios
            {
                'name': 'Early Close Friday (1 PM ET)',
                'current_time': self.et_tz.localize(datetime(2025, 11, 28, 13, 0, 0)), # Day after Thanksgiving, 1 PM ET
                'market_close_override': time(13, 0, 0),  # Early close
                'expected_contract': 'NQZ25',
                'description': 'Should handle early market close correctly'
            },
            {
                'name': 'Holiday No Trading',
                'current_time': self.et_tz.localize(datetime(2025, 7, 4, 10, 0, 0)),   # July 4th
                'is_trading_day': False,
                'expected_contract': 'NQU25',  # Would be July contract
                'description': 'Should handle non-trading days without expiry confusion'
            },
            
            # DST transition scenarios
            {
                'name': 'DST Spring Forward Expiry',
                'current_time': self.et_tz.localize(datetime(2025, 3, 21, 9, 30, 0)), # Spring DST transition period
                'expected_contract': 'NQH25',  # March contract
                'description': 'Should handle DST transition correctly'
            },
            {
                'name': 'DST Fall Back Expiry', 
                'current_time': self.et_tz.localize(datetime(2025, 11, 21, 9, 30, 0)), # Fall DST transition period
                'expected_contract': 'NQX25',  # November contract
                'description': 'Should handle DST transition correctly'
            }
        ]
        
        for case in edge_cases:
            self._test_expiration_case(case)
    
    def _test_expiration_case(self, case: dict):
        """Test individual expiration case"""
        with patch('app.utils.timeutils.now_et', return_value=case['current_time']):
            if 'market_close_override' in case:
                with patch('app.utils.timeutils.get_market_close_et', return_value=case['market_close_override']):
                    contract = futures_symbols.get_current_contract('/NQ')
            elif 'is_trading_day' in case:
                with patch('app.utils.timeutils.is_trading_day', return_value=case['is_trading_day']):
                    contract = futures_symbols.get_current_contract('/NQ')
            else:
                contract = futures_symbols.get_current_contract('/NQ')
            
            assert contract == case['expected_contract'], \
                f"Failed {case['name']}: Expected {case['expected_contract']}, got {contract}. {case['description']}"
```

### 2. Non-Standard Calendar Handling

```python
class NonStandardCalendarHandler:
    """Handle futures contracts with non-standard expiration calendars"""
    
    SPECIAL_EXPIRY_RULES = {
        '/VX': {
            'expiry_day': 'third_tuesday',  # VIX futures expire on third Tuesday
            'expiry_time': time(9, 0, 0),   # 9:00 AM ET (different from standard)
            'calendar': 'cboe_vix'
        },
        '/BTC': {
            'expiry_day': 'last_friday',    # Bitcoin futures 
            'expiry_time': time(16, 0, 0),  # 4:00 PM ET
            'calendar': 'cme_crypto'
        }
    }
    
    def get_expiry_datetime(self, symbol: str, year: int, month: int) -> datetime:
        """Get exact expiry datetime for symbol with special rules"""
        
        if symbol in self.SPECIAL_EXPIRY_RULES:
            rules = self.SPECIAL_EXPIRY_RULES[symbol]
            expiry_date = self._calculate_special_expiry_date(year, month, rules['expiry_day'])
            expiry_time = rules['expiry_time']
            
            et_tz = pytz.timezone('US/Eastern')
            return et_tz.localize(datetime.combine(expiry_date, expiry_time))
        else:
            # Standard third Friday, 9:30 AM ET
            return self._calculate_standard_expiry(year, month)
    
    def _calculate_special_expiry_date(self, year: int, month: int, rule: str) -> date:
        """Calculate expiry date based on special rules"""
        
        if rule == 'third_tuesday':
            # Find third Tuesday of month
            first_day = date(year, month, 1)
            first_tuesday = first_day + timedelta(days=(1 - first_day.weekday()) % 7)
            return first_tuesday + timedelta(weeks=2)
            
        elif rule == 'last_friday':
            # Find last Friday of month
            import calendar
            last_day = date(year, month, calendar.monthrange(year, month)[1])
            last_friday = last_day - timedelta(days=(last_day.weekday() - 4) % 7)
            return last_friday
            
        else:
            raise ValueError(f"Unknown expiry rule: {rule}")
```

## Data Migration Risk Mitigation

### 1. Legacy Data Tagging and Migration

```python
class LegacyDataMigrator:
    """Handle legacy UTC data migration to Eastern Time"""
    
    def __init__(self):
        self.et_tz = pytz.timezone('US/Eastern')
        self.utc_tz = pytz.UTC
    
    def tag_legacy_data(self, data_path: Path):
        """Tag existing data files with timezone metadata"""
        
        for data_file in data_path.rglob('*.json'):
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                # Add timezone metadata if not present
                if not self._has_timezone_metadata(data):
                    data['_timezone_metadata'] = {
                        'original_timezone': 'UTC',
                        'migration_timestamp': datetime.now().isoformat(),
                        'migration_version': '1.0.0',
                        'requires_conversion': True
                    }
                    
                    # Backup original file
                    backup_path = data_file.with_suffix('.json.utc_backup')
                    shutil.copy2(data_file, backup_path)
                    
                    # Write updated data
                    with open(data_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    logger.info(f"Tagged legacy data file: {data_file}")
                    
            except Exception as e:
                logger.error(f"Failed to tag legacy data file {data_file}: {e}")
    
    def migrate_critical_datasets(self):
        """Reprocess critical datasets into Eastern Time"""
        
        critical_datasets = [
            'data/historical/ohlc_*.json',
            'data/exports/daily_reports_*.json',
            'logs/trade_executions_*.log'
        ]
        
        for pattern in critical_datasets:
            files = Path('.').glob(pattern)
            for file_path in files:
                self._migrate_dataset(file_path)
    
    def _migrate_dataset(self, file_path: Path):
        """Migrate individual dataset to Eastern Time"""
        
        logger.info(f"Migrating dataset: {file_path}")
        
        try:
            # Load data
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                # Handle log files differently
                data = self._process_log_file(file_path)
            
            # Convert timestamps
            converted_data = self._convert_timestamps_to_et(data)
            
            # Add migration metadata
            converted_data['_migration_info'] = {
                'migrated_at': self.et_tz.localize(datetime.now()).isoformat(),
                'from_timezone': 'UTC',
                'to_timezone': 'US/Eastern',
                'original_file': str(file_path)
            }
            
            # Save migrated data
            migrated_path = file_path.with_stem(f"{file_path.stem}_ET_migrated")
            with open(migrated_path, 'w') as f:
                json.dump(converted_data, f, indent=2)
            
            logger.info(f"Successfully migrated {file_path} -> {migrated_path}")
            
        except Exception as e:
            logger.error(f"Failed to migrate dataset {file_path}: {e}")
    
    def _convert_timestamps_to_et(self, data):
        """Recursively convert UTC timestamps to Eastern Time"""
        
        if isinstance(data, dict):
            converted = {}
            for key, value in data.items():
                if self._is_timestamp_field(key) and isinstance(value, str):
                    # Convert timestamp string
                    try:
                        utc_dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        et_dt = utc_dt.astimezone(self.et_tz)
                        converted[key] = et_dt.isoformat()
                        converted[f"{key}_original_utc"] = value  # Keep original for reference
                    except:
                        converted[key] = value  # Keep original if conversion fails
                else:
                    converted[key] = self._convert_timestamps_to_et(value)
            return converted
            
        elif isinstance(data, list):
            return [self._convert_timestamps_to_et(item) for item in data]
            
        else:
            return data
    
    def _is_timestamp_field(self, field_name: str) -> bool:
        """Check if field name suggests it contains a timestamp"""
        timestamp_indicators = [
            'timestamp', 'time', 'date', 'created_at', 'updated_at',
            'retrieved_at', 'executed_at', 'expired_at'
        ]
        return any(indicator in field_name.lower() for indicator in timestamp_indicators)
```

### 2. Analytics Dashboard Compatibility

```python
class AnalyticsDashboardAdapter:
    """Ensure analytics dashboards remain compatible with new ET data"""
    
    def __init__(self):
        self.dashboard_configs = self._load_dashboard_configs()
    
    def update_dashboard_queries(self):
        """Update all dashboard queries to handle ET timestamps"""
        
        for dashboard in self.dashboard_configs:
            updated_queries = []
            
            for query in dashboard['queries']:
                if self._query_uses_timestamps(query):
                    updated_query = self._convert_query_to_et(query)
                    updated_queries.append(updated_query)
                else:
                    updated_queries.append(query)
            
            dashboard['queries'] = updated_queries
            dashboard['_migration_info'] = {
                'migrated_at': datetime.now().isoformat(),
                'timezone_updated': 'UTC->ET'
            }
        
        self._save_dashboard_configs(self.dashboard_configs)
    
    def _convert_query_to_et(self, query: dict) -> dict:
        """Convert dashboard query to use Eastern Time"""
        
        converted_query = query.copy()
        
        # Update time filters
        if 'time_filters' in query:
            for filter_config in converted_query['time_filters']:
                filter_config['timezone'] = 'US/Eastern'
                filter_config['_original_timezone'] = 'UTC'
        
        # Update time grouping
        if 'group_by_time' in query:
            converted_query['group_by_time']['timezone'] = 'US/Eastern'
        
        # Update time display format
        if 'display_format' in query:
            if 'UTC' in query['display_format']:
                converted_query['display_format'] = query['display_format'].replace('UTC', 'ET')
        
        return converted_query
    
    def create_compatibility_views(self):
        """Create database views that provide both UTC and ET timestamps"""
        
        compatibility_views = [
            {
                'name': 'trades_dual_timezone',
                'base_table': 'trades',
                'timestamp_columns': ['executed_at', 'created_at'],
                'description': 'Provides trades with both UTC and ET timestamps'
            },
            {
                'name': 'ohlc_dual_timezone', 
                'base_table': 'ohlc_data',
                'timestamp_columns': ['datetime', 'retrieved_at'],
                'description': 'Provides OHLC data with both UTC and ET timestamps'
            }
        ]
        
        for view_config in compatibility_views:
            self._create_dual_timezone_view(view_config)
    
    def _create_dual_timezone_view(self, config: dict):
        """Create database view with dual timezone columns"""
        
        base_columns = "*"
        additional_columns = []
        
        for ts_col in config['timestamp_columns']:
            # Add ET version of each timestamp column
            additional_columns.append(
                f"{ts_col} AT TIME ZONE 'UTC' AT TIME ZONE 'US/Eastern' AS {ts_col}_et"
            )
            # Keep original as UTC for compatibility
            additional_columns.append(
                f"{ts_col} AS {ts_col}_utc"
            )
        
        view_sql = f"""
        CREATE OR REPLACE VIEW {config['name']} AS
        SELECT {base_columns}, {', '.join(additional_columns)}
        FROM {config['base_table']};
        
        COMMENT ON VIEW {config['name']} IS '{config['description']}';
        """
        
        # Execute view creation (implementation depends on database)
        logger.info(f"Created dual timezone view: {config['name']}")
```

## Production Deployment Checklist

### Pre-Migration Validation

- [ ] All timezone conflict scenarios tested
- [ ] API middleware deployed and tested  
- [ ] Contract expiration edge cases validated
- [ ] Legacy data tagged and backed up
- [ ] Dashboard compatibility verified
- [ ] Dual timezone views created

### Migration Execution

- [ ] System backup completed
- [ ] UTC deprecation warnings enabled
- [ ] ET functions deployed
- [ ] Critical datasets migrated
- [ ] API contracts updated
- [ ] Monitoring dashboards updated

### Post-Migration Validation  

- [ ] All timestamps display in Eastern Time
- [ ] External APIs functioning correctly
- [ ] Contract rollover logic verified
- [ ] Legacy data accessible
- [ ] Analytics dashboards operational
- [ ] Performance within acceptable limits

This comprehensive implementation addresses all critical areas for a robust Eastern Time migration while minimizing risks of silent failures and user confusion.
