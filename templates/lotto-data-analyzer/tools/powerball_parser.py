"""
Powerball Data Parser
--------------------
Parses authentic Powerball data from text files into CSV format.
"""

import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Optional
import os

class PowerballParser:
    """Parser for Powerball text data files."""
    
    def __init__(self):
        self.draws = []
    
    def parse_file(self, file_path: str) -> List[Dict]:
        """
        Parse Powerball data from text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of drawing dictionaries
        """
        self.draws = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for date patterns
            if self._is_date_line(line):
                draw = self._parse_draw_block(lines, i)
                if draw:
                    self.draws.append(draw)
                    
            i += 1
        
        print(f"Parsed {len(self.draws)} drawings from {file_path}")
        return self.draws
    
    def _is_date_line(self, line: str) -> bool:
        """Check if line contains a date."""
        date_patterns = [
            r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\w+\s+\d{1,2},\s+\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\w+\s+\d{1,2},\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, line):
                return True
        return False
    
    def _parse_draw_block(self, lines: List[str], start_idx: int) -> Optional[Dict]:
        """Parse a single draw block starting from date line."""
        date_line = lines[start_idx].strip()
        
        # Parse the date
        draw_date = self._parse_date(date_line)
        if not draw_date:
            return None
        
        # Look for the 6 numbers following the date
        numbers = []
        i = start_idx + 1
        
        while i < len(lines) and len(numbers) < 6:
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Stop if we hit another date or "Power Play"
            if self._is_date_line(line) or line.startswith('Power Play'):
                break
            
            # Try to parse as number
            try:
                num = int(line)
                if 1 <= num <= 69:  # Valid number range
                    numbers.append(num)
            except ValueError:
                pass
            
            i += 1
        
        # We need exactly 6 numbers (5 white balls + 1 powerball)
        if len(numbers) != 6:
            return None
        
        # Validate the numbers
        white_balls = sorted(numbers[:5])  # First 5 are white balls (sort them)
        powerball = numbers[5]  # Last one is powerball
        
        # Check ranges
        if not all(1 <= n <= 69 for n in white_balls):
            return None
        if not (1 <= powerball <= 26):
            return None
        
        # Validate it's a proper Powerball drawing day
        draw_datetime = datetime.strptime(draw_date, '%m/%d/%Y')
        if draw_datetime.weekday() not in [0, 2, 5]:  # Mon, Wed, Sat
            return None
        
        return {
            'draw_date': draw_date,
            'n1': white_balls[0],
            'n2': white_balls[1],
            'n3': white_balls[2],
            'n4': white_balls[3],
            'n5': white_balls[4],
            'powerball': powerball
        }
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse various date formats and return MM/DD/YYYY."""
        date_patterns = [
            # "Mon, May 31, 2025"
            (r'(\w+),\s+(\w+)\s+(\d{1,2}),\s+(\d{4})', self._parse_long_date),
            # "05/31/2025" or "5/31/2025"
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', self._parse_slash_date),
            # "May 31, 2025"
            (r'(\w+)\s+(\d{1,2}),\s+(\d{4})', self._parse_month_date)
        ]
        
        for pattern, parser in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                return parser(match)
        
        return None
    
    def _parse_long_date(self, match) -> str:
        """Parse 'Mon, May 31, 2025' format."""
        _, month_name, day, year = match.groups()
        
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'June': 6, 'July': 7, 'August': 8, 'September': 9,
            'October': 10, 'November': 11, 'December': 12
        }
        
        month_num = month_map.get(month_name)
        if month_num:
            return f"{month_num:02d}/{int(day):02d}/{year}"
        return None
    
    def _parse_slash_date(self, match) -> str:
        """Parse '05/31/2025' format."""
        month, day, year = match.groups()
        return f"{int(month):02d}/{int(day):02d}/{year}"
    
    def _parse_month_date(self, match) -> str:
        """Parse 'May 31, 2025' format."""
        month_name, day, year = match.groups()
        
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'June': 6, 'July': 7, 'August': 8, 'September': 9,
            'October': 10, 'November': 11, 'December': 12
        }
        
        month_num = month_map.get(month_name)
        if month_num:
            return f"{month_num:02d}/{int(day):02d}/{year}"
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert parsed draws to DataFrame."""
        if not self.draws:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.draws)
        
        # Sort by date (newest first)
        df['draw_date'] = pd.to_datetime(df['draw_date'])
        df = df.sort_values('draw_date', ascending=False).reset_index(drop=True)
        
        # Convert date back to string format
        df['draw_date'] = df['draw_date'].dt.strftime('%m/%d/%Y')
        
        return df
    
    def save_to_csv(self, output_path: str) -> str:
        """Save parsed data to CSV file."""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(df)} draws to {output_path}")
        print(f"Date range: {df['draw_date'].iloc[0]} to {df['draw_date'].iloc[-1]}")
        
        return output_path
    
    def get_stats(self) -> Dict:
        """Get statistics about the parsed data."""
        if not self.draws:
            return {'error': 'No data parsed'}
        
        df = self.to_dataframe()
        df['draw_date'] = pd.to_datetime(df['draw_date'])
        
        return {
            'total_draws': len(df),
            'date_range': {
                'start': df['draw_date'].min().strftime('%Y-%m-%d'),
                'end': df['draw_date'].max().strftime('%Y-%m-%d')
            },
            'white_ball_range': {
                'min': df[['n1', 'n2', 'n3', 'n4', 'n5']].min().min(),
                'max': df[['n1', 'n2', 'n3', 'n4', 'n5']].max().max()
            },
            'powerball_range': {
                'min': df['powerball'].min(),
                'max': df['powerball'].max()
            }
        }

def parse_powerball_file(input_path: str, output_path: str = None) -> str:
    """
    Parse Powerball text file and save as CSV.
    
    Args:
        input_path: Path to input text file
        output_path: Path for output CSV (optional)
        
    Returns:
        Path to the created CSV file
    """
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"data/{base_name}_parsed.csv"
    
    parser = PowerballParser()
    parser.parse_file(input_path)
    
    # Show statistics
    stats = parser.get_stats()
    print("\n=== PARSING RESULTS ===")
    print(f"Total draws parsed: {stats['total_draws']:,}")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"White balls: {stats['white_ball_range']['min']}-{stats['white_ball_range']['max']}")
    print(f"Powerball: {stats['powerball_range']['min']}-{stats['powerball_range']['max']}")
    
    return parser.save_to_csv(output_path)

if __name__ == "__main__":
    # Parse the provided Powerball file
    csv_file = parse_powerball_file(
        "attached_assets/powerball-2025-1999.txt",
        "data/powerball_complete_dataset.csv"
    )
    print(f"\nComplete dataset saved as: {csv_file}")