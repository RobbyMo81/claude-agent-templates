# Set paths
$projectRoot = "C:\Users\RobMo\OneDrive\Documents\helios"
$archivePath = "C:\Users\RobMo\OneDrive\Documents\helios\archive"
$dryRun = $true  # Set to $false to execute move and generate README

# Define backend files to preserve
$preserve = @(
    "server.py",
    "agent.py",
    "trainer.py",
    "memory_store.py",
    "metacognition.py",
    "decision_engine.py",
    "cross_model_analytics.py",
    "requirements.txt",
    "environment.yml",
    ".venv",  # or "venv"
    ".git",
    ".gitignore"
)

# Normalize preserve paths
$preserveFull = $preserve | ForEach-Object { Join-Path $projectRoot $_ }

# Get all items in project root
$allItems = Get-ChildItem -Path $projectRoot -Force

# Function to check if item should be preserved
function IsPreserved($item) {
    foreach ($preserved in $preserveFull) {
        if ($item.FullName -eq $preserved -or $item.FullName.StartsWith($preserved + "\")) {
            return $true
        }
    }
    return $false
}

# Create archive folder if needed
if (-not (Test-Path $archivePath)) {
    if (-not $dryRun) {
        New-Item -ItemType Directory -Path $archivePath | Out-Null
    }
    Write-Host "[INFO] Archive folder created: $archivePath"
}

# Initialize manifest content
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$manifest = @()
$manifest += "# Helios Archive Manifest"
$manifest += ""
$manifest += "Archived on: $timestamp"
$manifest += ""
$manifest += "## Moved Files and Folders"
$manifest += ""

# Move non-preserved items and build manifest
foreach ($item in $allItems) {
    if (-not (IsPreserved $item)) {
        $destination = Join-Path $archivePath $item.Name
        if ($dryRun) {
            Write-Host "[DRY RUN] Would move: $($item.FullName) → $destination"
            $manifest += "- $($item.Name)"
        } else {
            try {
                Move-Item -Path $item.FullName -Destination $destination -Force
                Write-Host "[MOVED] $($item.FullName) → $destination"
                $manifest += "- $($item.Name)"
            } catch {
                Write-Host "[ERROR] Failed to move: $($item.FullName) — $($_.Exception.Message)"
            }
        }
    } else {
        Write-Host ("[KEPT] " + $item.FullName)
    }
}

# Write manifest to README.md in archive folder
$readmePath = Join-Path $archivePath "README.md"
if (-not $dryRun) {
    $manifest | Out-File -FilePath $readmePath -Encoding UTF8
    Write-Host "[INFO] Manifest written to: $readmePath"
} else {
    Write-Host "[DRY RUN] Manifest would be written to: $readmePath"
}