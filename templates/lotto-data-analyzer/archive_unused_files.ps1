<#
.SYNOPSIS
    Archives unused files and organizes markdown files in the LottoDataAnalyzer project.
.DESCRIPTION
    Creates an archive directory for unused files while maintaining directory structure.
    Moves all markdown files to a unified markdown folder.
    Generates a log file with details of the operations.
.NOTES
    Author: GitHub Copilot
    Date: June 24, 2025
#>

# Base directories
$projectRoot = $PSScriptRoot
$archiveRoot = Join-Path $projectRoot "archive"
$markdownRoot = Join-Path $projectRoot "docs\markdown"
$logFile = Join-Path $projectRoot "maintenance_log.md"

# Create directories if they don't exist
if (-not (Test-Path $archiveRoot)) {
    New-Item -ItemType Directory -Path $archiveRoot | Out-Null
    Write-Host "Created archive directory: $archiveRoot" -ForegroundColor Green
}

if (-not (Test-Path $markdownRoot)) {
    New-Item -ItemType Directory -Path $markdownRoot -Force | Out-Null
    Write-Host "Created unified markdown directory: $markdownRoot" -ForegroundColor Green
}

# List of files to archive with their relative paths
$filesToArchive = @(
    "core\data_prep_legacy.py",
    "core\ml_legacy.py",
    "core\ml_prediction_interface_old.py",
    "core\utils_deprecated.py",
    "core\experimental\early_prototypes.py",
    "scripts\legacy_data_import.py",
    "notebooks\prototype_models.ipynb",
    "templates\old_ui_components.html",
    "core\visualizers\deprecated_charts.py",
    "core\storage_v1.py"
)

# Initialize log content
$logContent = @"
# Maintenance Operation Log

**Date:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Part 1: Files Archived

The following files were identified as unused and moved to the archive directory:

| Original Path | Archive Path | Status | Reason |
|---------------|-------------|--------|--------|
"@

# Counter for archived files
$archivedCount = 0
$notFoundCount = 0
$skippedCount = 0

# Process each file to archive
foreach ($relativeFilePath in $filesToArchive) {
    $originalFilePath = Join-Path $projectRoot $relativeFilePath
    $archiveFilePath = Join-Path $archiveRoot $relativeFilePath
    $archiveDir = Split-Path -Parent $archiveFilePath
    
    # Status and reason variables
    $status = ""
    $reason = ""
    
    # Check if file exists
    if (Test-Path $originalFilePath) {
        try {
            # Create directory structure if needed
            if (-not (Test-Path $archiveDir)) {
                New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
            }
            
            # Copy file to archive
            Copy-Item -Path $originalFilePath -Destination $archiveFilePath -Force
            
            # Check if copy was successful
            if (Test-Path $archiveFilePath) {
                # Remove original file
                Remove-Item -Path $originalFilePath -Force
                $status = "✅ Archived"
                $reason = "Unused file identified during system health review"
                $archivedCount++
                Write-Host "Archived: $relativeFilePath" -ForegroundColor Green
            } else {
                $status = "❌ Failed"
                $reason = "Copy operation failed"
                $skippedCount++
                Write-Host "Failed to archive: $relativeFilePath" -ForegroundColor Red
            }
        } catch {
            $status = "❌ Error"
            $reason = $_.Exception.Message
            $skippedCount++
            Write-Host "Error archiving: $relativeFilePath - $($_.Exception.Message)" -ForegroundColor Red
        }
    } else {
        $status = "⚠️ Not Found"
        $reason = "Original file does not exist"
        $notFoundCount++
        Write-Host "File not found: $relativeFilePath" -ForegroundColor Yellow
    }
    
    # Add to log
    $logContent += "`n| $relativeFilePath | $relativeFilePath | $status | $reason |"
}

# Add archive summary to log
$logContent += @"

### Archive Operation Summary

- **Total files processed:** $($filesToArchive.Count)
- **Successfully archived:** $archivedCount
- **Files not found:** $notFoundCount
- **Files skipped/errors:** $skippedCount

## Part 2: Markdown File Organization

The following markdown files were moved to the unified markdown directory:

| Original Path | New Path | Status |
|---------------|----------|--------|
"@

# Find all markdown files in the project (excluding the archive directory)
$markdownFiles = Get-ChildItem -Path $projectRoot -Recurse -File -Filter "*.md" | 
                 Where-Object { $_.FullName -notlike "$archiveRoot*" -and $_.FullName -notlike "$markdownRoot*" }

# Counter for moved markdown files
$movedCount = 0
$markdownSkippedCount = 0

# Process each markdown file
foreach ($mdFile in $markdownFiles) {
    # Get relative path from project root
    $relativePath = $mdFile.FullName.Substring($projectRoot.Length + 1)
    
    # Create new file name (preserve directory structure in the filename to avoid collisions)
    $dirStructure = (Split-Path -Parent $relativePath) -replace '\\', '-'
    if ($dirStructure) {
        $newFileName = "$dirStructure-$($mdFile.Name)"
    } else {
        $newFileName = $mdFile.Name
    }
    
    # Clean up new filename (remove multiple hyphens, etc.)
    $newFileName = $newFileName -replace '-+', '-'
    
    # Create destination path
    $destinationPath = Join-Path $markdownRoot $newFileName
    
    # Move the file
    try {
        Copy-Item -Path $mdFile.FullName -Destination $destinationPath -Force
        
        # Check if copy was successful
        if (Test-Path $destinationPath) {
            # Remove original file
            Remove-Item -Path $mdFile.FullName -Force
            $status = "✅ Moved"
            $movedCount++
            Write-Host "Moved markdown file: $relativePath -> $newFileName" -ForegroundColor Green
        } else {
            $status = "❌ Failed"
            $markdownSkippedCount++
            Write-Host "Failed to move markdown file: $relativePath" -ForegroundColor Red
        }
    } catch {
        $status = "❌ Error"
        $markdownSkippedCount++
        Write-Host "Error moving markdown file: $relativePath - $($_.Exception.Message)" -ForegroundColor Red
    }
    
    # Add to log
    $logContent += "`n| $relativePath | $newFileName | $status |"
}

# Add markdown summary to log
$logContent += @"

### Markdown Organization Summary

- **Total markdown files processed:** $($markdownFiles.Count)
- **Successfully moved:** $movedCount
- **Files skipped/errors:** $markdownSkippedCount

## Recommendations

1. **Review the unified markdown folder** to ensure all documentation is properly organized
2. **Update any references** to the moved markdown files in your code
3. **Run all tests** to ensure system functionality is not affected by archived files
4. **Consider keeping the archive** for at least one release cycle

To restore an archived file, copy it from the archive directory back to its original location.
"@

# Write log file
Set-Content -Path $logFile -Value $logContent

# Create a simple index.md file in the markdown folder
$indexContent = @"
# Markdown Documentation Index

This folder contains all markdown documentation for the LottoDataAnalyzer project.

## Available Documents

$(($markdownFiles | ForEach-Object { "- [$($_.Name)]($($_.Name))" }) -join "`n")

*This index was automatically generated on $(Get-Date -Format "yyyy-MM-dd") during the documentation reorganization.*
"@

Set-Content -Path (Join-Path $markdownRoot "index.md") -Value $indexContent

# Display summary
Write-Host "`n---- Maintenance Operations Complete ----" -ForegroundColor Cyan
Write-Host "`n1. Archive Operation:" -ForegroundColor White
Write-Host "   Total files processed: $($filesToArchive.Count)" -ForegroundColor White
Write-Host "   Successfully archived: $archivedCount" -ForegroundColor Green
Write-Host "   Files not found: $notFoundCount" -ForegroundColor Yellow
Write-Host "   Files skipped/errors: $skippedCount" -ForegroundColor Red

Write-Host "`n2. Markdown Organization:" -ForegroundColor White
Write-Host "   Total markdown files processed: $($markdownFiles.Count)" -ForegroundColor White
Write-Host "   Successfully moved: $movedCount" -ForegroundColor Green
Write-Host "   Files skipped/errors: $markdownSkippedCount" -ForegroundColor Red

Write-Host "`nLog file created at: $logFile" -ForegroundColor Cyan
Write-Host "`nMarkdown index created at: $(Join-Path $markdownRoot "index.md")" -ForegroundColor Cyan
Write-Host "`nTo restore archived files, copy them from the archive directory back to their original locations." -ForegroundColor White