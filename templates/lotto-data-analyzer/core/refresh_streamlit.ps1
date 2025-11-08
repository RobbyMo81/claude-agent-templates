# Define the Streamlit script path
$streamlitScript = "C:\Users\RobMo\OneDrive\Documents\LottoDataAnalyzer\app.py"

# Kill any existing Streamlit processes
Get-Process | Where-Object { $_.Name -like "python" -and $_.Path -like "*streamlit*" } | Stop-Process -Force

# Wait for a moment to ensure the process is stopped
Start-Sleep -Seconds 2

# Start the Streamlit server again
Start-Process -NoNewWindow -FilePath "streamlit" -ArgumentList "run $streamlitScript"
