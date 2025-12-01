"""
Official Powerball Data Scraper
------------------------------
Fetches authentic Powerball drawing results from official sources.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import trafilatura
import re
import json
from typing import List, Dict, Optional
import time

class PowerballScraper:
    """Scraper for official Powerball drawing results."""
    
    def __init__(self):
        self.base_url = "https://www.powerball.com"
        self.api_url = "https://www.powerball.com/api/v1/numbers"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_recent_draws(self, days_back: int = 365) -> List[Dict]:
        """
        Fetch recent Powerball drawings from official API.
        
        Args:
            days_back: Number of days back to fetch
            
        Returns:
            List of drawing dictionaries
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Format dates for API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            print(f"Fetching Powerball data from {start_str} to {end_str}...")
            
            # Try API endpoint first
            api_url = f"{self.api_url}?start={start_str}&end={end_str}"
            response = self.session.get(api_url, timeout=30)
            
            if response.status_code == 200:
                return self._parse_api_response(response.json())
            else:
                print(f"API failed with status {response.status_code}, trying web scraping...")
                return self._scrape_web_results(days_back)
                
        except Exception as e:
            print(f"Error fetching recent draws: {e}")
            return []
    
    def _parse_api_response(self, data: Dict) -> List[Dict]:
        """Parse API response into standardized format."""
        draws = []
        
        if 'draws' in data:
            for draw in data['draws']:
                try:
                    draw_data = {
                        'draw_date': draw.get('date'),
                        'n1': int(draw['numbers'][0]),
                        'n2': int(draw['numbers'][1]),
                        'n3': int(draw['numbers'][2]),
                        'n4': int(draw['numbers'][3]),
                        'n5': int(draw['numbers'][4]),
                        'powerball': int(draw['powerball'])
                    }
                    draws.append(draw_data)
                except (KeyError, ValueError, IndexError) as e:
                    print(f"Error parsing draw: {e}")
                    continue
        
        return draws
    
    def _scrape_web_results(self, days_back: int) -> List[Dict]:
        """Fallback web scraping method."""
        try:
            # Scrape the main results page
            results_url = f"{self.base_url}/previous-results"
            response = self.session.get(results_url, timeout=30)
            
            if response.status_code != 200:
                print(f"Failed to access results page: {response.status_code}")
                return []
            
            # Extract text content
            text_content = trafilatura.extract(response.text)
            if not text_content:
                print("Could not extract content from results page")
                return []
            
            # Parse the content for draw results
            draws = self._parse_results_text(text_content, days_back)
            return draws
            
        except Exception as e:
            print(f"Web scraping error: {e}")
            return []
    
    def _parse_results_text(self, content: str, days_back: int) -> List[Dict]:
        """Parse text content to extract draw results."""
        draws = []
        
        # Look for date and number patterns
        # Pattern: Date followed by 5 numbers and powerball
        date_pattern = r'(\d{1,2}/\d{1,2}/\d{4})'
        number_pattern = r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})'
        
        lines = content.split('\n')
        current_date = None
        
        for line in lines:
            line = line.strip()
            
            # Check for date
            date_match = re.search(date_pattern, line)
            if date_match:
                current_date = date_match.group(1)
                continue
            
            # Check for numbers on same line or next line
            if current_date:
                number_match = re.search(number_pattern, line)
                if number_match:
                    try:
                        numbers = [int(x) for x in number_match.groups()]
                        if len(numbers) == 6:
                            draw_data = {
                                'draw_date': current_date,
                                'n1': numbers[0],
                                'n2': numbers[1], 
                                'n3': numbers[2],
                                'n4': numbers[3],
                                'n5': numbers[4],
                                'powerball': numbers[5]
                            }
                            draws.append(draw_data)
                            current_date = None  # Reset for next draw
                    except ValueError:
                        continue
        
        return draws
    
    def validate_draws(self, draws: List[Dict]) -> List[Dict]:
        """Validate that draws have correct number ranges and dates."""
        valid_draws = []
        
        for draw in draws:
            try:
                # Check date format
                draw_date = datetime.strptime(draw['draw_date'], '%m/%d/%Y')
                
                # Check if it's a valid Powerball day (Mon, Wed, Sat)
                if draw_date.weekday() not in [0, 2, 5]:
                    print(f"Invalid draw day: {draw['draw_date']} ({draw_date.strftime('%A')})")
                    continue
                
                # Check number ranges
                white_balls = [draw['n1'], draw['n2'], draw['n3'], draw['n4'], draw['n5']]
                if not all(1 <= n <= 69 for n in white_balls):
                    print(f"Invalid white ball numbers: {white_balls}")
                    continue
                
                if not (1 <= draw['powerball'] <= 26):
                    print(f"Invalid powerball number: {draw['powerball']}")
                    continue
                
                # Check for sorted white balls (should be ascending)
                if white_balls != sorted(white_balls):
                    draw['n1'], draw['n2'], draw['n3'], draw['n4'], draw['n5'] = sorted(white_balls)
                
                valid_draws.append(draw)
                
            except (ValueError, KeyError) as e:
                print(f"Error validating draw: {e}")
                continue
        
        return valid_draws
    
    def fetch_and_update_dataset(self, existing_csv: str, days_back: int = 1095) -> str:
        """
        Fetch recent draws and update existing dataset.
        
        Args:
            existing_csv: Path to existing CSV file
            days_back: How many days back to fetch (default: 3 years)
            
        Returns:
            Path to updated CSV file
        """
        print("Fetching authentic Powerball data from official sources...")
        
        # Fetch new draws
        new_draws = self.get_recent_draws(days_back)
        
        if not new_draws:
            print("No new draws found from official sources")
            return existing_csv
        
        # Validate the draws
        valid_draws = self.validate_draws(new_draws)
        print(f"Found {len(valid_draws)} valid draws from official sources")
        
        if not valid_draws:
            print("No valid draws to add")
            return existing_csv
        
        # Load existing data
        try:
            existing_df = pd.read_csv(existing_csv)
            existing_df['draw_date'] = pd.to_datetime(existing_df['draw_date'])
        except Exception as e:
            print(f"Error loading existing data: {e}")
            return existing_csv
        
        # Convert new draws to DataFrame
        new_df = pd.DataFrame(valid_draws)
        new_df['draw_date'] = pd.to_datetime(new_df['draw_date'])
        
        # Find the most recent date in existing data
        latest_existing = existing_df['draw_date'].max()
        print(f"Latest existing draw: {latest_existing.strftime('%Y-%m-%d')}")
        
        # Filter new draws to only include those after latest existing
        newer_draws = new_df[new_df['draw_date'] > latest_existing]
        
        if len(newer_draws) == 0:
            print("No newer draws found - dataset is up to date")
            return existing_csv
        
        print(f"Adding {len(newer_draws)} new draws to dataset")
        
        # Combine and sort
        combined_df = pd.concat([newer_draws, existing_df], ignore_index=True)
        combined_df = combined_df.sort_values('draw_date', ascending=False).reset_index(drop=True)
        
        # Remove any duplicates based on date
        combined_df = combined_df.drop_duplicates(subset=['draw_date'], keep='first')
        
        # Format dates back to string
        combined_df['draw_date'] = combined_df['draw_date'].dt.strftime('%m/%d/%Y')
        
        # Save updated dataset
        updated_path = existing_csv.replace('.csv', '_updated.csv')
        combined_df.to_csv(updated_path, index=False)
        
        print(f"Updated dataset saved to: {updated_path}")
        print(f"Total records: {len(combined_df)}")
        print(f"Date range: {combined_df['draw_date'].iloc[0]} to {combined_df['draw_date'].iloc[-1]}")
        
        return updated_path

def update_powerball_data():
    """Main function to update Powerball data."""
    scraper = PowerballScraper()
    
    # Update the corrected dataset
    updated_path = scraper.fetch_and_update_dataset(
        'data/powerball_history_corrected.csv', 
        days_back=1460  # 4 years back to ensure we get everything since 2021
    )
    
    return updated_path

if __name__ == "__main__":
    update_powerball_data()