# Windows Stress Test Troubleshooting Guide

This guide helps resolve common issues when running stress tests on Windows.

## Quick Start

The simplest way to run stress tests on Windows is:

```bash
npm run stress-test
```

This will run the Node.js-based stress test runner which is the most reliable across Windows environments.

## Alternative Commands

If you encounter issues, try these alternatives in order:

### 1. Simple Node.js Runner (Recommended)
```bash
npm run stress-test:simple
```

### 2. PowerShell Version
```bash
npm run stress-test:powershell
```

### 3. Batch File Version
```bash
npm run stress-test:windows
```

### 4. Direct Node.js Execution
```bash
node tests/runStressTest.js
```

## Common Issues and Solutions

### Issue: "stress-test.bat is not recognized"

**Solution:** The batch file might not be in the correct directory or there's a PATH issue.

**Fix:**
```bash
# Run directly from the project root
cmd /c stress-test.bat

# Or use the simple Node.js version
npm run stress-test:simple
```

### Issue: PowerShell execution policy errors

**Solution:** Windows blocks PowerShell script execution by default.

**Fix:**
```powershell
# Temporary fix (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or use bypass for single execution
powershell -ExecutionPolicy Bypass -File stress-test.ps1
```

### Issue: "npm ERR! missing script"

**Solution:** The npm script might not be defined correctly.

**Fix:**
```bash
# Check available scripts
npm run

# Use direct Node.js execution
node tests/runStressTest.js
```

### Issue: Node.js not found

**Solution:** Node.js is not installed or not in PATH.

**Fix:**
1. Download and install Node.js from [nodejs.org](https://nodejs.org/)
2. Restart your terminal/command prompt
3. Verify with: `node --version`

### Issue: Docker commands fail

**Solution:** Docker Desktop is not running or not installed.

**Fix:**
1. Install Docker Desktop for Windows
2. Start Docker Desktop
3. Verify with: `docker --version`

### Issue: Permission denied errors

**Solution:** Windows file permissions or antivirus blocking execution.

**Fix:**
1. Run terminal as Administrator
2. Add project folder to antivirus exceptions
3. Use Node.js version: `node tests/runStressTest.js`

## Manual Testing

If all automated tests fail, you can manually verify the system:

### 1. Check Project Structure
```bash
# Verify key files exist
dir services\config.ts
dir docker-compose.yml
dir Dockerfile
```

### 2. Test Node.js and Dependencies
```bash
node --version
npm --version
npm list
```

### 3. Test Docker
```bash
docker --version
docker-compose --version
docker-compose config
```

### 4. Test TypeScript Compilation
```bash
npx tsc --noEmit --skipLibCheck
```

### 5. Test Build Process
```bash
npm run build
```

## Environment-Specific Notes

### Windows 10/11 with WSL
If you have WSL (Windows Subsystem for Linux) installed:
```bash
# Use the Unix version in WSL
wsl bash stress-test.sh
```

### Windows with Git Bash
If you're using Git Bash:
```bash
# Use the Unix shell script
bash stress-test.sh
```

### Command Prompt vs PowerShell
- **Command Prompt**: Use `.bat` files and `npm run stress-test:windows`
- **PowerShell**: Use `.ps1` files and `npm run stress-test:powershell`
- **Both**: Use Node.js directly with `npm run stress-test:simple`

## Getting Help

If you continue to have issues:

1. **Check the terminal output** for specific error messages
2. **Try the simple Node.js version** first: `npm run stress-test:simple`
3. **Verify prerequisites**:
   - Node.js 16+ installed
   - npm working correctly
   - Project dependencies installed (`npm install`)
4. **Run a basic test**:
   ```bash
   node -e "console.log('Node.js works!'); process.exit(0)"
   ```

## Success Indicators

A successful stress test run should show:
- Multiple "[SUCCESS]" messages
- A results summary with high success rate
- "ALL STRESS TESTS PASSED" or similar positive message
- Exit code 0 (or low failure count)

If you see mostly failures, check:
- Are all project files in place?
- Is Docker running (for Docker tests)?
- Are dependencies installed (`npm install`)?

## Quick Recovery

To reset and try again:
```bash
# Clean and reinstall dependencies
npm clean-install

# Use the most compatible test runner
npm run stress-test:simple

# Check specific components
node -e "console.log('Testing:', require('fs').existsSync('services/config.ts') ? ' Config exists' : ' Config missing')"
```
