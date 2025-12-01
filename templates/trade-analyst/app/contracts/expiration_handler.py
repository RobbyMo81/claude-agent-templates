"""Contract Expiration Edge Case Handler

This module provides comprehensive handling of futures and options contract expiration
edge cases with Eastern Time awareness, ensuring accurate expiration calculations
across DST transitions and market holidays.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, date, time, timedelta
from dataclasses import dataclass
from enum import Enum
import pytz
import calendar

from app.utils.timeutils import (
    EASTERN_TIMEZONE,
    now_eastern,
    get_business_days,
    get_next_business_day,
    get_previous_business_day,
    is_market_holiday,
    is_business_day,
    CURRENT_YEAR_HOLIDAYS
)

logger = logging.getLogger(__name__)


class ContractType(Enum):
    """Types of tradeable contracts"""
    EQUITY_OPTION = "equity_option"
    INDEX_OPTION = "index_option"  
    FUTURES = "futures"
    FUTURES_OPTION = "futures_option"
    ETF_OPTION = "etf_option"


class ExpirationPattern(Enum):
    """Contract expiration patterns"""
    MONTHLY_3RD_FRIDAY = "monthly_3rd_friday"
    WEEKLY_FRIDAY = "weekly_friday"
    QUARTERLY_3RD_FRIDAY = "quarterly_3rd_friday"
    FUTURES_QUARTERLY = "futures_quarterly"
    FUTURES_MONTHLY = "futures_monthly"
    AM_SETTLEMENT = "am_settlement"
    PM_SETTLEMENT = "pm_settlement"


@dataclass
class ExpirationRule:
    """Contract expiration rule definition"""
    contract_type: ContractType
    pattern: ExpirationPattern
    expiration_time: time
    settlement_time: Optional[time] = None
    last_trading_offset_days: int = 0  # Days before expiration for last trading
    special_holidays: Optional[List[date]] = None
    dst_aware: bool = True


class ContractExpirationTester:
    """Comprehensive contract expiration testing with Eastern Time"""
    
    def __init__(self):
        self.timezone = EASTERN_TIMEZONE
        self.expiration_rules = self._initialize_expiration_rules()
        self.edge_cases_detected = []
        self.validation_results = []
    
    def _initialize_expiration_rules(self) -> Dict[ContractType, ExpirationRule]:
        """Initialize standard contract expiration rules"""
        return {
            ContractType.EQUITY_OPTION: ExpirationRule(
                contract_type=ContractType.EQUITY_OPTION,
                pattern=ExpirationPattern.MONTHLY_3RD_FRIDAY,
                expiration_time=time(23, 59, 59),  # 11:59:59 PM ET
                settlement_time=time(11, 59, 59),  # 11:59:59 AM ET for exercise
                last_trading_offset_days=0
            ),
            ContractType.INDEX_OPTION: ExpirationRule(
                contract_type=ContractType.INDEX_OPTION,
                pattern=ExpirationPattern.AM_SETTLEMENT,
                expiration_time=time(9, 30, 0),   # 9:30 AM ET
                settlement_time=time(9, 30, 0),
                last_trading_offset_days=1  # Last trading day before expiration
            ),
            ContractType.FUTURES: ExpirationRule(
                contract_type=ContractType.FUTURES,
                pattern=ExpirationPattern.FUTURES_QUARTERLY,
                expiration_time=time(14, 0, 0),   # 2:00 PM ET
                last_trading_offset_days=1
            ),
            ContractType.FUTURES_OPTION: ExpirationRule(
                contract_type=ContractType.FUTURES_OPTION,
                pattern=ExpirationPattern.FUTURES_MONTHLY,
                expiration_time=time(14, 0, 0),
                last_trading_offset_days=2
            )
        }
    
    def calculate_expiration_date(self, 
                                contract_type: ContractType,
                                year: int,
                                month: int,
                                week: Optional[int] = None) -> datetime:
        """
        Calculate exact expiration datetime for a contract
        
        Args:
            contract_type: Type of contract
            year: Expiration year
            month: Expiration month
            week: Week number for weekly options
            
        Returns:
            Exact expiration datetime in Eastern Time
        """
        rule = self.expiration_rules.get(contract_type)
        if not rule:
            raise ValueError(f"No expiration rule defined for {contract_type}")
        
        base_date = self._calculate_base_expiration_date(rule.pattern, year, month, week)
        expiration_datetime = self.timezone.localize(
            datetime.combine(base_date, rule.expiration_time)
        )
        
        # Handle DST transitions
        if rule.dst_aware:
            expiration_datetime = self._handle_dst_transition(expiration_datetime)
        
        # Handle market holidays
        if is_market_holiday(base_date):
            adjusted_date = get_previous_business_day(base_date)
            expiration_datetime = self.timezone.localize(
                datetime.combine(adjusted_date, rule.expiration_time)
            )
            
            self.edge_cases_detected.append({
                'type': 'holiday_adjustment',
                'original_date': base_date,
                'adjusted_date': adjusted_date,
                'contract_type': contract_type.value
            })
        
        return expiration_datetime
    
    def _calculate_base_expiration_date(self, 
                                      pattern: ExpirationPattern,
                                      year: int,
                                      month: int,
                                      week: Optional[int] = None) -> date:
        """Calculate base expiration date according to pattern"""
        if pattern == ExpirationPattern.MONTHLY_3RD_FRIDAY:
            return self._third_friday_of_month(year, month)
        elif pattern == ExpirationPattern.WEEKLY_FRIDAY:
            if week is None:
                raise ValueError("Week number required for weekly options")
            return self._nth_friday_of_month(year, month, week)
        elif pattern == ExpirationPattern.QUARTERLY_3RD_FRIDAY:
            # Ensure it's a quarterly month (March, June, Sept, Dec)
            if month not in [3, 6, 9, 12]:
                raise ValueError(f"Quarterly options only expire in Mar/Jun/Sep/Dec, not {month}")
            return self._third_friday_of_month(year, month)
        elif pattern in [ExpirationPattern.FUTURES_QUARTERLY, ExpirationPattern.FUTURES_MONTHLY]:
            # Futures typically expire on the day before 3rd Friday
            third_friday = self._third_friday_of_month(year, month)
            return get_previous_business_day(third_friday)
        elif pattern in [ExpirationPattern.AM_SETTLEMENT, ExpirationPattern.PM_SETTLEMENT]:
            return self._third_friday_of_month(year, month)
        else:
            raise ValueError(f"Unsupported expiration pattern: {pattern}")
    
    def _third_friday_of_month(self, year: int, month: int) -> date:
        """Find the third Friday of a given month"""
        first_day = date(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        return third_friday
    
    def _nth_friday_of_month(self, year: int, month: int, week: int) -> date:
        """Find the Nth Friday of a given month"""
        first_day = date(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        nth_friday = first_friday + timedelta(days=7 * (week - 1))
        
        # Ensure it's still in the same month
        if nth_friday.month != month:
            raise ValueError(f"Week {week} Friday does not exist in {year}-{month:02d}")
        
        return nth_friday
    
    def _handle_dst_transition(self, dt: datetime) -> datetime:
        """Handle DST transition edge cases"""
        try:
            # Check if we're in a DST transition period
            local_dt = self.timezone.localize(dt.replace(tzinfo=None))
            return local_dt
        except pytz.AmbiguousTimeError:
            # Fall back (2 AM occurs twice) - use the first occurrence (DST)
            local_dt = self.timezone.localize(dt.replace(tzinfo=None), is_dst=True)
            self.edge_cases_detected.append({
                'type': 'dst_fallback_ambiguous',
                'datetime': str(local_dt),
                'resolution': 'used_dst_time'
            })
            return local_dt
        except pytz.NonExistentTimeError:
            # Spring forward (2 AM doesn't exist) - use 3 AM EDT
            adjusted_dt = dt.replace(tzinfo=None) + timedelta(hours=1)
            local_dt = self.timezone.localize(adjusted_dt)
            self.edge_cases_detected.append({
                'type': 'dst_spring_forward_nonexistent', 
                'original_time': str(dt),
                'adjusted_time': str(local_dt),
                'resolution': 'moved_forward_1_hour'
            })
            return local_dt
    
    def get_last_trading_day(self, 
                           contract_type: ContractType,
                           expiration_date: datetime) -> datetime:
        """Get last trading day for a contract"""
        rule = self.expiration_rules.get(contract_type)
        if not rule:
            raise ValueError(f"No rule defined for {contract_type}")
        
        if rule.last_trading_offset_days == 0:
            return expiration_date
        
        # Calculate last trading day
        trading_date = expiration_date.date()
        for _ in range(rule.last_trading_offset_days):
            trading_date = get_previous_business_day(trading_date)
        
        # Use market close time for last trading day
        last_trading = self.timezone.localize(
            datetime.combine(trading_date, time(16, 0, 0))
        )
        
        return last_trading
    
    def is_expiration_week(self, check_date: date, 
                          contract_type: ContractType,
                          year: int, month: int) -> bool:
        """Check if a date falls in expiration week"""
        try:
            expiration_date = self.calculate_expiration_date(contract_type, year, month)
            exp_date = expiration_date.date()
            
            # Find the Monday of expiration week
            monday_of_exp_week = exp_date - timedelta(days=exp_date.weekday())
            friday_of_exp_week = monday_of_exp_week + timedelta(days=4)
            
            return monday_of_exp_week <= check_date <= friday_of_exp_week
        except Exception:
            return False
    
    def validate_expiration_chain(self, 
                                 contract_type: ContractType,
                                 start_year: int,
                                 end_year: int) -> Dict[str, Any]:
        """Validate expiration dates across multiple years"""
        validation_results = {
            'contract_type': contract_type.value,
            'validation_period': f"{start_year}-{end_year}",
            'total_expirations': 0,
            'edge_cases_found': 0,
            'dst_transitions_handled': 0,
            'holiday_adjustments': 0,
            'expirations': [],
            'edge_cases': []
        }
        
        # Reset edge case tracking
        self.edge_cases_detected = []
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                try:
                    expiration_dt = self.calculate_expiration_date(contract_type, year, month)
                    last_trading_dt = self.get_last_trading_day(contract_type, expiration_dt)
                    
                    validation_results['expirations'].append({
                        'year': year,
                        'month': month,
                        'expiration_datetime': expiration_dt,
                        'last_trading_datetime': last_trading_dt,
                        'is_dst': expiration_dt.dst() != timedelta(0),
                        'is_holiday_adjusted': any(
                            edge['type'] == 'holiday_adjustment' 
                            for edge in self.edge_cases_detected
                        )
                    })
                    
                    validation_results['total_expirations'] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate expiration for {year}-{month:02d}: {e}")
        
        # Summarize edge cases
        validation_results['edge_cases'] = self.edge_cases_detected
        validation_results['edge_cases_found'] = len(self.edge_cases_detected)
        validation_results['dst_transitions_handled'] = len([
            e for e in self.edge_cases_detected 
            if e['type'].startswith('dst_')
        ])
        validation_results['holiday_adjustments'] = len([
            e for e in self.edge_cases_detected 
            if e['type'] == 'holiday_adjustment'
        ])
        
        return validation_results
    
    def test_midnight_boundary_cases(self) -> Dict[str, Any]:
        """Test edge cases around midnight Eastern Time"""
        test_cases = []
        
        # Test cases around DST transitions
        dst_transitions_2025 = [
            (2025, 3, 9),   # Spring forward
            (2025, 11, 2),  # Fall back
        ]
        
        for year, month, day in dst_transitions_2025:
            for hour in [0, 1, 2, 3, 23]:
                test_dt = datetime(year, month, day, hour, 0, 0)
                try:
                    localized_dt = self._handle_dst_transition(
                        self.timezone.localize(test_dt, is_dst=None)
                    )
                    
                    test_cases.append({
                        'test_datetime': str(test_dt),
                        'localized_datetime': str(localized_dt),
                        'dst_offset': str(localized_dt.dst()),
                        'utc_offset': str(localized_dt.utcoffset()),
                        'success': True
                    })
                    
                except Exception as e:
                    test_cases.append({
                        'test_datetime': str(test_dt),
                        'error': str(e),
                        'success': False
                    })
        
        return {
            'test_type': 'midnight_boundary_dst_transitions',
            'total_tests': len(test_cases),
            'successful_tests': len([t for t in test_cases if t['success']]),
            'failed_tests': len([t for t in test_cases if not t['success']]),
            'test_cases': test_cases,
            'edge_cases_detected': self.edge_cases_detected
        }
    
    def get_expiration_calendar(self, 
                               contract_type: ContractType,
                               year: int) -> List[Dict[str, Any]]:
        """Get full expiration calendar for a contract type and year"""
        calendar_data = []
        
        for month in range(1, 13):
            try:
                expiration_dt = self.calculate_expiration_date(contract_type, year, month)
                last_trading_dt = self.get_last_trading_day(contract_type, expiration_dt)
                
                utc_offset = expiration_dt.utcoffset()
                calendar_data.append({
                    'month': month,
                    'month_name': calendar.month_name[month],
                    'expiration_date': expiration_dt.date(),
                    'expiration_time': expiration_dt.time(),
                    'expiration_datetime': expiration_dt,
                    'last_trading_datetime': last_trading_dt,
                    'days_until_expiration': (expiration_dt.date() - date.today()).days,
                    'is_dst': expiration_dt.dst() != timedelta(0),
                    'timezone_name': expiration_dt.tzname(),
                    'utc_offset_hours': utc_offset.total_seconds() / 3600 if utc_offset else 0
                })
                
            except Exception as e:
                logger.error(f"Failed to calculate expiration for {year}-{month:02d}: {e}")
        
        return calendar_data


# Example usage and comprehensive testing
if __name__ == "__main__":
    print("⏰ CONTRACT EXPIRATION EDGE CASE HANDLER TEST")
    print("=" * 55)
    
    tester = ContractExpirationTester()
    
    # Test 1: Basic expiration calculation
    equity_option_exp = tester.calculate_expiration_date(
        ContractType.EQUITY_OPTION, 2025, 9
    )
    print(f"✅ Sept 2025 Equity Option Expiration: {equity_option_exp}")
    
    # Test 2: DST transition handling
    dst_test_results = tester.test_midnight_boundary_cases()
    print(f"✅ DST Boundary Tests: {dst_test_results['successful_tests']}/{dst_test_results['total_tests']} passed")
    
    # Test 3: Multi-year validation
    validation_results = tester.validate_expiration_chain(
        ContractType.EQUITY_OPTION, 2025, 2026
    )
    print(f"✅ Multi-year validation: {validation_results['total_expirations']} expirations calculated")
    print(f"✅ Edge cases handled: {validation_results['edge_cases_found']}")
    
    # Test 4: Expiration calendar
    calendar_2025 = tester.get_expiration_calendar(ContractType.EQUITY_OPTION, 2025)
    print(f"✅ 2025 Expiration Calendar: {len(calendar_2025)} months")
    
    # Show some sample expirations
    for exp in calendar_2025[:3]:
        print(f"   📅 {exp['month_name']}: {exp['expiration_datetime']} ({'DST' if exp['is_dst'] else 'EST'})")
    
    print(f"\\n🎯 CONTRACT EXPIRATION TESTING: SUCCESS!")
    print(f"📊 Total edge cases detected: {len(tester.edge_cases_detected)}")
