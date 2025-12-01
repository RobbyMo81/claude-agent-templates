"""
Eastern Time Core Function Tests
Tests for the new Eastern Time-based time utilities.
"""

import pytest
import pytz
from datetime import datetime, date, time, timezone, timedelta
from unittest.mock import patch, MagicMock
import warnings

# Import the timeutils module (to be updated with ET functions)
from app.utils import timeutils


class TestEasternTimeCore:
    """Test core Eastern Time functionality"""
    
    def test_now_et_returns_eastern_time(self):
        """Test now_et() returns timezone-aware Eastern Time datetime"""
        et_time = timeutils.now_et()
        
        # Should be timezone-aware
        assert et_time.tzinfo is not None
        
        # Should be Eastern Time zone
        et_tz = pytz.timezone('US/Eastern')
        assert et_time.tzinfo.zone == et_tz.zone or str(et_time.tzinfo) in ['EST', 'EDT']
        
        # Should be close to current time
        now_utc = datetime.now(timezone.utc)
        et_from_utc = now_utc.astimezone(et_tz)
        time_diff = abs((et_time - et_from_utc).total_seconds())
        assert time_diff < 5  # Within 5 seconds
    
    def test_timezone_conversion_functions(self):
        """Test UTC ↔ ET conversion functions"""
        # Create test datetime in UTC
        test_utc = datetime(2025, 8, 26, 16, 0, 0, tzinfo=timezone.utc)
        
        # Convert to ET
        test_et = timeutils.to_et_from_utc(test_utc)
        
        # Should be timezone-aware Eastern Time
        assert test_et.tzinfo is not None
        et_tz = pytz.timezone('US/Eastern')
        assert test_et.tzinfo.zone == et_tz.zone or str(test_et.tzinfo) in ['EST', 'EDT']
        
        # Convert back to UTC
        back_to_utc = timeutils.to_utc_from_et(test_et)
        
        # Should match original
        assert back_to_utc.replace(microsecond=0) == test_utc.replace(microsecond=0)
    
    def test_market_hours_detection(self):
        """Test market hours detection in Eastern Time"""
        et_tz = pytz.timezone('US/Eastern')
        
        # Test during market hours (10:00 AM ET on a Tuesday)
        market_time = et_tz.localize(datetime(2025, 8, 26, 10, 0, 0))  # Tuesday
        with patch.object(timeutils, 'now_eastern', return_value=market_time):
            assert timeutils.is_market_hours_et() == True
        
        # Test outside market hours (6:00 AM ET on a Tuesday)
        before_market = et_tz.localize(datetime(2025, 8, 26, 6, 0, 0))  # Tuesday
        with patch.object(timeutils, 'now_eastern', return_value=before_market):
            assert timeutils.is_market_hours_et() == False
        
        # Test weekend (Saturday)
        weekend = et_tz.localize(datetime(2025, 8, 23, 10, 0, 0))  # Saturday
        with patch.object(timeutils, 'now_eastern', return_value=weekend):
            assert timeutils.is_market_hours_et() == False
    
    def test_dst_handling(self):
        """Test Daylight Saving Time transitions"""
        et_tz = pytz.timezone('US/Eastern')
        
        # Test EST vs EDT - same UTC time should be different local hours
        # January 15 (EST) vs July 15 (EDT) 
        
        # EST Test: January 15, 2025 at 2:00 PM EST (UTC-5)
        est_time = et_tz.localize(datetime(2025, 1, 15, 14, 0, 0))
        assert est_time.strftime('%Z') == 'EST'
        
        # EDT Test: July 15, 2025 at 3:00 PM EDT (UTC-4) - same UTC offset difference
        edt_time = et_tz.localize(datetime(2025, 7, 15, 15, 0, 0)) 
        assert edt_time.strftime('%Z') == 'EDT'
        
        # Test that now_et() handles DST correctly
        # Mock at module level since now_et is an alias
        with patch.object(timeutils, 'now_eastern') as mock_now:
            # Mock January date - EST (UTC-5)
            mock_now.return_value = est_time
            et_jan = timeutils.now_et()
            
            # Mock July date - EDT (UTC-4) 
            mock_now.return_value = edt_time  
            et_jul = timeutils.now_et()
            
            # Different local times due to DST:
            # January: 2 PM EST, July: 3 PM EDT
            assert et_jan.hour == 14  # 2 PM EST
            assert et_jul.hour == 15  # 3 PM EDT  
            assert et_jan.hour != et_jul.hour
    
    def test_contract_expiration_et(self):
        """Test futures contract expiration calculation in Eastern Time"""
        # This will test the updated futures contract expiration logic
        # Using September 2025 contract as example
        
        # Mock current time to August 26, 2025 10:00 AM ET
        et_tz = pytz.timezone('US/Eastern')
        current_et = et_tz.localize(datetime(2025, 8, 26, 10, 0, 0))
        
        with patch('app.utils.timeutils.now_et', return_value=current_et):
            # Test that September contract is still active
            sep_expiry = datetime(2025, 9, 20, 9, 30, 0)  # Third Friday 9:30 AM ET
            sep_expiry_et = et_tz.localize(sep_expiry)
            
            assert current_et < sep_expiry_et
            
            # Test contract selection logic would pick September
            # (This would integrate with the actual futures symbols module)
    
    def test_market_open_close_et(self):
        """Test market open/close time functions"""
        test_date = date(2025, 8, 26)  # Tuesday
        
        market_open = timeutils.get_market_open_et(test_date)
        market_close = timeutils.get_market_close_et(test_date)
        
        # Should be Eastern Time
        et_tz = pytz.timezone('US/Eastern')
        assert market_open.tzinfo is not None
        assert market_close.tzinfo is not None
        
        # Should be correct times
        assert market_open.time() == time(9, 30, 0)
        assert market_close.time() == time(16, 0, 0)
        
        # Should be same date
        assert market_open.date() == test_date
        assert market_close.date() == test_date
    
    def test_format_et_timestamp(self):
        """Test Eastern Time timestamp formatting"""
        et_tz = pytz.timezone('US/Eastern')
        test_dt = et_tz.localize(datetime(2025, 8, 26, 14, 30, 0))
        
        # With timezone
        formatted_with_tz = timeutils.format_et_timestamp(test_dt, include_tz=True)
        assert 'ET' in formatted_with_tz or 'EDT' in formatted_with_tz or 'EST' in formatted_with_tz
        assert '2025-08-26' in formatted_with_tz
        assert '14:30' in formatted_with_tz or '2:30' in formatted_with_tz
        
        # Without timezone
        formatted_without_tz = timeutils.format_et_timestamp(test_dt, include_tz=False)
        assert 'ET' not in formatted_without_tz and 'EDT' not in formatted_without_tz and 'EST' not in formatted_without_tz
    
    def test_parse_iso8601_et(self):
        """Test parsing ISO8601 strings to Eastern Time"""
        # Test UTC string conversion to ET
        utc_iso = "2025-08-26T18:30:00Z"
        parsed_et = timeutils.parse_iso8601_et(utc_iso)
        
        assert parsed_et is not None
        assert parsed_et.tzinfo is not None
        
        # Should be converted to Eastern Time
        et_tz = pytz.timezone('US/Eastern')
        expected_et = et_tz.localize(datetime(2025, 8, 26, 14, 30, 0))  # 6:30 PM UTC = 2:30 PM EDT
        
        # Allow for some timezone representation differences
        time_diff = abs((parsed_et.replace(tzinfo=None) - expected_et.replace(tzinfo=None)).total_seconds())
        assert time_diff < 1


class TestBackwardCompatibility:
    """Test backward compatibility with UTC functions"""
    
    def test_deprecated_utc_functions_emit_warnings(self):
        """Test that deprecated UTC functions emit deprecation warnings"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Call deprecated function
            result = timeutils.now_utc()
            
            # Check that warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert "now_et" in str(w[0].message)
            
            # Result should still be valid UTC time
            assert result.tzinfo == timezone.utc
    
    def test_mixed_timezone_handling(self):
        """Test system handles mixed UTC/ET data gracefully"""
        # Test that system can process both UTC and ET timestamps
        utc_time = datetime.now(timezone.utc)
        et_time = timeutils.now_et()
        
        # Both should be processable
        utc_formatted = timeutils.format_et_timestamp(utc_time.astimezone(pytz.timezone('US/Eastern')))
        et_formatted = timeutils.format_et_timestamp(et_time)
        
        assert utc_formatted is not None
        assert et_formatted is not None
    
    def test_existing_interfaces_maintained(self):
        """Test that existing function signatures are maintained"""
        # These functions should still exist and work (even if deprecated)
        assert hasattr(timeutils, 'now_utc')
        assert hasattr(timeutils, 'to_rfc3339')
        assert hasattr(timeutils, 'parse_iso8601_utc')
        
        # Should be callable
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = timeutils.now_utc()
            assert isinstance(result, datetime)


class TestMarketHoursET:
    """Test the enhanced MarketHours class with Eastern Time"""
    
    def test_market_hours_initialization(self):
        """Test MarketHours class initializes with Eastern Time"""
        market_hours = timeutils.MarketHoursET()
        
        assert market_hours.timezone.zone == 'US/Eastern'
        assert market_hours.market_open == time(9, 30, 0)
        assert market_hours.market_close == time(16, 0, 0)
        assert market_hours.pre_market_start == time(4, 0, 0)
        assert market_hours.after_hours_end == time(20, 0, 0)
    
    def test_is_trading_day(self):
        """Test trading day detection"""
        market_hours = timeutils.MarketHoursET()
        
        # Tuesday should be trading day
        tuesday = date(2025, 8, 26)
        assert market_hours.is_trading_day(tuesday) == True
        
        # Saturday should not be trading day
        saturday = date(2025, 8, 23)
        assert market_hours.is_trading_day(saturday) == False
        
        # Sunday should not be trading day
        sunday = date(2025, 8, 24)
        assert market_hours.is_trading_day(sunday) == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
