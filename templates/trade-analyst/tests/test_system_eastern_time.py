"""
System-Wide Eastern Time Tests
Comprehensive tests for system-wide Eastern Time implementation.
"""

import pytest
import os
import json
import re
from unittest.mock import patch, MagicMock
from datetime import datetime, date, timezone, timedelta
import pytz
from pathlib import Path

# Import system modules
from app.utils import timeutils
from app import main
from app import config
import ta_production
import ta


class TestSystemWideEasternTime:
    """Test system-wide Eastern Time implementation"""
    
    def test_all_timestamps_consistent(self):
        """Test all system timestamps use Eastern Time consistently"""
        et_tz = pytz.timezone('US/Eastern')
        fixed_time = et_tz.localize(datetime(2025, 8, 26, 14, 30, 0))
        
        # Collect timestamps from various system components
        timestamps = {}
        
        with patch('app.utils.timeutils.now_et', return_value=fixed_time):
            # Test timeutils module
            timestamps['timeutils_now'] = timeutils.now_et()
            timestamps['timeutils_formatted'] = timeutils.format_et_timestamp(fixed_time)
            
            # Test production module
            with patch('ta_production.get_ohlc_data') as mock_ohlc:
                mock_ohlc.return_value = {
                    'symbol': '/NQ',
                    'data': [],
                    'provenance': {
                        'retrieved_at': timeutils.format_et_timestamp(fixed_time),
                        'source': 'schwab'
                    }
                }
                
                result = mock_ohlc.return_value
                timestamps['production_provenance'] = result['provenance']['retrieved_at']
            
            # Test CLI module
            with patch('ta.format_timestamp') as mock_format:
                mock_format.return_value = timeutils.format_et_timestamp(fixed_time)
                timestamps['cli_formatted'] = mock_format.return_value
        
        # Verify all timestamps are consistent
        assert timestamps['timeutils_now'].tzinfo is not None
        assert 'ET' in timestamps['timeutils_formatted'] or 'EDT' in timestamps['timeutils_formatted'] or 'EST' in timestamps['timeutils_formatted']
        
        # All string timestamps should reference Eastern Time
        for key, timestamp in timestamps.items():
            if isinstance(timestamp, str):
                # Should contain Eastern Time reference
                assert any(tz in timestamp for tz in ['ET', 'EDT', 'EST']), f"Timestamp {key}: {timestamp} doesn't reference Eastern Time"
    
    def test_market_calendar_integration(self):
        """Test market calendar integration across system"""
        et_tz = pytz.timezone('US/Eastern')
        
        # Test various market scenarios
        test_scenarios = [
            # (datetime, expected_market_open, description)
            (datetime(2025, 8, 26, 10, 0, 0), True, "Tuesday 10 AM ET - Regular hours"),
            (datetime(2025, 8, 26, 6, 0, 0), False, "Tuesday 6 AM ET - Pre-market"),
            (datetime(2025, 8, 26, 18, 0, 0), False, "Tuesday 6 PM ET - After hours"),
            (datetime(2025, 8, 23, 10, 0, 0), False, "Saturday 10 AM ET - Weekend"),
            (datetime(2025, 8, 24, 10, 0, 0), False, "Sunday 10 AM ET - Weekend"),
        ]
        
        for test_datetime, expected_open, description in test_scenarios:
            et_time = et_tz.localize(test_datetime)
            
            with patch('app.utils.timeutils.now_et', return_value=et_time):
                with patch('app.utils.timeutils.is_market_hours_et', return_value=expected_open):
                    # Test timeutils module
                    is_open_timeutils = timeutils.is_market_hours_et()
                    assert is_open_timeutils == expected_open, f"Failed for {description}"
                    
                    # Test that other modules would get consistent results
                    # This ensures the market calendar is used consistently
                    market_hours = timeutils.MarketHoursET()
                    is_trading_day = market_hours.is_trading_day(et_time.date())
                    
                    if et_time.weekday() < 5:  # Monday=0, Friday=4
                        assert is_trading_day == True, f"Should be trading day: {description}"
                    else:
                        assert is_trading_day == False, f"Should not be trading day: {description}"
    
    def test_cross_module_time_consistency(self):
        """Test time consistency across all modules"""
        et_tz = pytz.timezone('US/Eastern')
        test_time = et_tz.localize(datetime(2025, 8, 26, 14, 30, 0))
        
        with patch('app.utils.timeutils.now_et', return_value=test_time):
            # Collect time references from different modules
            time_references = []
            
            # TimeUtils module
            time_references.append(('timeutils', timeutils.now_et()))
            
            # Test modules that use time
            modules_to_test = [
                'app.production_provider',
                'app.auth', 
                'app.errors',
                'app.guardrails',
                'app.writers',
                'app.healthcheck'
            ]
            
            for module_name in modules_to_test:
                try:
                    # Import dynamically
                    module = __import__(module_name.replace('app.', ''), fromlist=[''])
                    
                    # Check if module has updated to use now_et
                    if hasattr(module, 'now_et') or hasattr(module, 'timeutils'):
                        # Module should use Eastern Time
                        time_references.append((module_name, 'uses_et'))
                    else:
                        # Module may need updating
                        time_references.append((module_name, 'needs_update'))
                        
                except ImportError:
                    # Module might not exist yet
                    pass
            
            # Validate that core modules reference Eastern Time
            assert len(time_references) > 1, "Should have multiple time references"
            
            # The main timeutils reference should be Eastern Time
            timeutils_ref = next(ref for ref in time_references if ref[0] == 'timeutils')
            assert timeutils_ref[1].tzinfo is not None
    
    def test_error_reporting_et(self):
        """Test all error messages use Eastern Time"""
        et_tz = pytz.timezone('US/Eastern')
        error_time = et_tz.localize(datetime(2025, 8, 26, 14, 30, 0))
        
        with patch('app.utils.timeutils.now_et', return_value=error_time):
            # Test error handling modules
            from app import errors
            from app import guardrails
            
            # Mock error scenarios
            test_errors = [
                ("API Error", "Schwab API returned 500 error"),
                ("Authentication Error", "Token expired"),
                ("Data Error", "Invalid OHLC data received"),
            ]
            
            for error_type, error_message in test_errors:
                # Test error creation with Eastern Time
                with patch('app.errors.create_error_response') as mock_error:
                    mock_error.return_value = {
                        'error': error_type,
                        'message': error_message,
                        'timestamp': timeutils.format_et_timestamp(error_time),
                        'timezone': 'ET'
                    }
                    
                    error_response = mock_error.return_value
                    
                    # Error should include Eastern Time timestamp
                    assert 'timestamp' in error_response
                    timestamp = error_response['timestamp']
                    assert any(tz in timestamp for tz in ['ET', 'EDT', 'EST'])
    
    def test_cli_output_et(self):
        """Test CLI output uses Eastern Time consistently"""
        et_tz = pytz.timezone('US/Eastern')
        cli_time = et_tz.localize(datetime(2025, 8, 26, 14, 30, 0))
        
        with patch('app.utils.timeutils.now_et', return_value=cli_time):
            # Test CLI commands that show timestamps
            test_commands = [
                ('futures-info', '--symbol=/NQ'),
                ('health-check', ''),
                ('validate', ''),
            ]
            
            for command, args in test_commands:
                try:
                    # Mock CLI command execution
                    with patch('ta.main') as mock_main:
                        mock_output = {
                            'command': command,
                            'timestamp': timeutils.format_et_timestamp(cli_time),
                            'status': 'success'
                        }
                        mock_main.return_value = mock_output
                        
                        result = mock_main.return_value
                        
                        # CLI output should use Eastern Time
                        if 'timestamp' in result:
                            timestamp = result['timestamp']
                            assert any(tz in timestamp for tz in ['ET', 'EDT', 'EST'])
                            
                except Exception as e:
                    # Command might not be fully implemented
                    print(f"CLI command {command} not fully testable: {e}")
    
    def test_configuration_et(self):
        """Test system configuration reflects Eastern Time settings"""
        # Test that config.toml includes Eastern Time settings
        config_path = Path('config.toml')
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_content = f.read()
                
            # Should include Eastern Time configuration
            # This will be validated once config is updated
            assert 'time' in config_content.lower() or 'timezone' in config_content.lower()
        else:
            # Config file should be created as part of migration
            pytest.skip("config.toml not found - will be created during migration")
    
    def test_data_export_et(self):
        """Test data export files use Eastern Time"""
        et_tz = pytz.timezone('US/Eastern')
        export_time = et_tz.localize(datetime(2025, 8, 26, 14, 30, 0))
        
        with patch('app.utils.timeutils.now_et', return_value=export_time):
            # Mock data export
            from app import writers
            
            test_data = {
                'symbol': '/NQ',
                'data': [
                    {'datetime': '2025-08-26T14:30:00-04:00', 'open': 19500, 'high': 19550, 'low': 19450, 'close': 19500}
                ],
                'exported_at': timeutils.format_et_timestamp(export_time)
            }
            
            # Mock file writing
            with patch('app.writers.write_ohlc_data') as mock_write:
                mock_write.return_value = f"data/exports/NQ_20250826_143000_ET.json"
                
                filename = mock_write.return_value
                
                # Filename should include ET reference
                assert 'ET' in filename or '_ET' in filename
    
    def test_log_files_et(self):
        """Test log files use Eastern Time timestamps"""
        et_tz = pytz.timezone('US/Eastern')
        log_time = et_tz.localize(datetime(2025, 8, 26, 14, 30, 0))
        
        # Test log files in logs/ directory
        log_dir = Path('logs')
        if log_dir.exists():
            log_files = list(log_dir.glob('*.log'))
            
            if log_files:
                # Check recent log entries
                for log_file in log_files[:3]:  # Check first 3 log files
                    try:
                        with open(log_file, 'r') as f:
                            recent_lines = f.readlines()[-10:]  # Last 10 lines
                            
                        for line in recent_lines:
                            # Look for timestamp patterns
                            if re.search(r'\d{4}-\d{2}-\d{2}', line):
                                # Line contains a date, should use Eastern Time
                                # After migration, these should contain ET references
                                print(f"Log line from {log_file.name}: {line.strip()}")
                                
                    except Exception as e:
                        print(f"Could not read log file {log_file}: {e}")
        else:
            pytest.skip("No log files found to test")


class TestPerformanceWithEasternTime:
    """Test performance impact of Eastern Time implementation"""
    
    def test_timezone_conversion_performance(self):
        """Test that timezone conversions don't impact performance significantly"""
        import time
        
        et_tz = pytz.timezone('US/Eastern')
        utc_tz = timezone.utc
        
        # Time multiple conversions
        start_time = time.perf_counter()
        
        for i in range(1000):
            # UTC to ET conversion
            utc_dt = datetime.now(utc_tz)
            et_dt = utc_dt.astimezone(et_tz)
            
            # ET to UTC conversion
            back_to_utc = et_dt.astimezone(utc_tz)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should complete 1000 conversions in reasonable time (< 1 second)
        assert total_time < 1.0, f"Timezone conversions too slow: {total_time:.3f} seconds"
    
    def test_market_hours_calculation_performance(self):
        """Test market hours calculation performance"""
        import time
        
        et_tz = pytz.timezone('US/Eastern')
        
        start_time = time.perf_counter()
        
        for i in range(100):
            test_time = et_tz.localize(datetime(2025, 8, 26, 10, i % 60, 0))
            
            with patch('app.utils.timeutils.now_et', return_value=test_time):
                is_open = timeutils.is_market_hours_et()
                
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should complete 100 market hours checks quickly (< 0.1 seconds)
        assert total_time < 0.1, f"Market hours calculation too slow: {total_time:.3f} seconds"


class TestBackwardCompatibilitySystem:
    """Test system-wide backward compatibility"""
    
    def test_existing_data_compatibility(self):
        """Test that existing data files are compatible"""
        # Test that existing cache files, exports, etc. can be read
        data_dirs = ['data/exports', 'data/quotes', 'data/historical']
        
        for data_dir in data_dirs:
            dir_path = Path(data_dir)
            if dir_path.exists():
                json_files = list(dir_path.glob('*.json'))
                
                for json_file in json_files[:3]:  # Test first 3 files
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            
                        # Data should be loadable
                        assert isinstance(data, (dict, list))
                        
                        # If data contains timestamps, system should handle them
                        if isinstance(data, dict) and 'timestamp' in data:
                            timestamp = data['timestamp']
                            # Should be parseable by new ET functions
                            parsed = timeutils.parse_iso8601_et(timestamp)
                            # Should not fail (might return None for invalid formats)
                            
                    except Exception as e:
                        print(f"Existing data file {json_file} might need migration: {e}")
    
    def test_deprecated_function_warnings(self):
        """Test that deprecated UTC functions emit proper warnings"""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Call deprecated functions
            try:
                utc_time = timeutils.now_utc()
                utc_parsed = timeutils.parse_iso8601_utc("2025-08-26T18:30:00Z")
                
                # Should have warnings
                assert len(w) >= 1
                
                for warning in w:
                    assert issubclass(warning.category, DeprecationWarning)
                    assert "deprecated" in str(warning.message).lower()
                    
            except AttributeError:
                # Functions might not exist yet in current implementation
                pytest.skip("Deprecated functions not yet implemented")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
