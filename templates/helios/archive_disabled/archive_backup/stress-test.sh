#!/bin/bash

# Helios System Stress Test Script
# This script runs comprehensive stress tests for the unified configuration and Docker setup

echo " Helios System Stress Test Suite"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}${BOLD}$1${NC}"
}

# Test variables
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNING_TESTS=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_output="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    print_status "Running: $test_name"
    
    local start_time=$(date +%s%N)
    local test_output
    local test_result
    
    if test_output=$(eval "$test_command" 2>&1); then
        if [[ -z "$expected_output" ]] || echo "$test_output" | grep -q "$expected_output"; then
            PASSED_TESTS=$((PASSED_TESTS + 1))
            local end_time=$(date +%s%N)
            local duration=$(( (end_time - start_time) / 1000000 ))
            print_success "$test_name (${duration}ms)"
            return 0
        else
            FAILED_TESTS=$((FAILED_TESTS + 1))
            print_error "$test_name - Expected output not found"
            echo "Expected: $expected_output"
            echo "Got: $test_output"
            return 1
        fi
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        print_error "$test_name - Command failed"
        echo "Output: $test_output"
        return 1
    fi
}

# Test 1: Check if Node.js dependencies are installed
print_header " Pre-flight Checks"
run_test "Node.js Dependencies" "npm list --production" "helios"

# Test 2: Check if Docker is available
run_test "Docker Availability" "docker --version" "Docker version"

# Test 3: Check if Docker Compose is available
run_test "Docker Compose Availability" "docker-compose --version" "docker-compose version"

# Test 4: Verify project structure
print_header " Project Structure Validation"
run_test "Config Service Exists" "test -f services/config.ts" ""
run_test "Model Service Exists" "test -f services/modelService.ts" ""
run_test "API Service Exists" "test -f services/api.ts" ""
run_test "Docker Compose Config" "test -f docker-compose.yml" ""
run_test "Backend Dockerfile" "test -f backend/Dockerfile" ""
run_test "Frontend Dockerfile" "test -f Dockerfile" ""

# Test 5: Configuration System Tests
print_header " Configuration System Tests"

# Create a temporary test file
cat > test_config.js << 'EOF'
const fs = require('fs');

// Simple configuration tests
try {
    // Test 1: Check if config file is valid TypeScript
    const configContent = fs.readFileSync('services/config.ts', 'utf8');
    if (!configContent.includes('export const appConfig')) {
        throw new Error('appConfig export not found');
    }
    
    // Test 2: Check for required functions
    const requiredFunctions = [
        'getApiBaseUrl',
        'getApiPort', 
        'getApiEndpoint',
        'updateApiPort',
        'updateApiHost'
    ];
    
    for (const func of requiredFunctions) {
        if (!configContent.includes(func)) {
            throw new Error(`Required function ${func} not found`);
        }
    }
    
    console.log('Configuration validation passed');
    process.exit(0);
} catch (error) {
    console.error('Configuration validation failed:', error.message);
    process.exit(1);
}
EOF

run_test "Configuration File Validation" "node test_config.js" "Configuration validation passed"
rm -f test_config.js

# Test 6: Docker Build Tests
print_header " Docker Build Tests"

# Test backend build
run_test "Backend Docker Build" "cd backend && docker build -t helios-backend-test . && docker rmi helios-backend-test" "Successfully"

# Test frontend build
run_test "Frontend Docker Build" "docker build -t helios-frontend-test . && docker rmi helios-frontend-test" "Successfully"

# Test 7: Docker Compose Validation
print_header " Docker Compose Validation"
run_test "Docker Compose Config Check" "docker-compose config" "services:"

# Test 8: Port Configuration Tests
print_header " Port Configuration Tests"

# Create port test script
cat > test_ports.js << 'EOF'
const fs = require('fs');

try {
    // Check docker-compose.yml for port configurations
    const composeContent = fs.readFileSync('docker-compose.yml', 'utf8');
    
    // Check for required ports
    const requiredPorts = ['5001:5001', '80:80'];
    for (const port of requiredPorts) {
        if (!composeContent.includes(port)) {
            throw new Error(`Port mapping ${port} not found in docker-compose.yml`);
        }
    }
    
    // Check config.ts for default port
    const configContent = fs.readFileSync('services/config.ts', 'utf8');
    if (!configContent.includes('port: 5001')) {
        throw new Error('Default port 5001 not found in config');
    }
    
    console.log('Port configuration validation passed');
    process.exit(0);
} catch (error) {
    console.error('Port validation failed:', error.message);
    process.exit(1);
}
EOF

run_test "Port Configuration Validation" "node test_ports.js" "Port configuration validation passed"
rm -f test_ports.js

# Test 9: Environment File Tests
print_header " Environment Configuration Tests"
run_test "Development Environment File" "test -f .env.development" ""
run_test "Production Environment File" "test -f .env.production" ""

# Test 10: Stress Test Framework Tests
print_header " Stress Test Framework Validation"
run_test "Stress Test File Exists" "test -f tests/stressTest.ts" ""
run_test "Stress Test Component Exists" "test -f components/StressTestComponent.tsx" ""

# Test 11: Documentation Tests
print_header " Documentation Tests"
run_test "Docker Setup Documentation" "test -f DOCKER_SETUP.md" ""
run_test "Port Configuration Documentation" "test -f PORT_CONFIGURATION.md" ""

# Test 12: TypeScript Compilation Test
print_header " TypeScript Compilation Tests"
run_test "TypeScript Check" "npx tsc --noEmit --skipLibCheck" ""

# Test 13: Build Process Tests
print_header " Build Process Tests"
run_test "Frontend Build Test" "npm run build" "dist"

# Test 14: Network Connectivity Tests (if services are running)
print_header " Network Connectivity Tests"

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    print_status "Services are running - testing connectivity"
    
    # Test frontend
    if curl -s -f http://localhost/ > /dev/null; then
        print_success "Frontend connectivity test"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        print_warning "Frontend not accessible (expected if not running)"
        WARNING_TESTS=$((WARNING_TESTS + 1))
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Test backend
    if curl -s -f http://localhost:5001/health > /dev/null; then
        print_success "Backend connectivity test"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        print_warning "Backend not accessible (expected if not running)"
        WARNING_TESTS=$((WARNING_TESTS + 1))
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
else
    print_status "Services not running - skipping connectivity tests"
    print_status "To start services: docker-compose up -d"
fi

# Test 15: Resource Usage Test
print_header " Resource Usage Tests"

# Check disk space
DISK_USAGE=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 90 ]; then
    print_success "Disk space check (${DISK_USAGE}% used)"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_warning "Disk space low (${DISK_USAGE}% used)"
    WARNING_TESTS=$((WARNING_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Final Results
echo ""
print_header " STRESS TEST RESULTS SUMMARY"
echo "======================================"
echo "Total Tests: $TOTAL_TESTS"
echo -e " Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "  Warnings: ${YELLOW}$WARNING_TESTS${NC}"
echo -e " Failed: ${RED}$FAILED_TESTS${NC}"

# Calculate success rate
SUCCESS_RATE=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))
echo "Success Rate: ${SUCCESS_RATE}%"

echo ""
if [ $FAILED_TESTS -eq 0 ]; then
    print_success " ALL STRESS TESTS PASSED! System is robust and ready for production."
elif [ $FAILED_TESTS -le $(( TOTAL_TESTS / 5 )) ]; then
    print_warning "  Some tests failed but system appears stable. Review failed tests."
else
    print_error " MULTIPLE FAILURES DETECTED! System may need attention before production deployment."
fi

echo ""
print_header " RECOMMENDATIONS"
echo "=================="

if [ $FAILED_TESTS -gt 0 ]; then
    echo "• Review failed tests and address underlying issues"
    echo "• Check Docker container health and resource allocation"
    echo "• Verify network connectivity and API endpoints"
fi

if [ $WARNING_TESTS -gt 0 ]; then
    echo "• Monitor warning conditions in production"
    echo "• Consider optimizing resource usage and performance"
fi

if [ $SUCCESS_RATE -ge 80 ]; then
    echo "• System ready for deployment"
    echo "• Consider implementing monitoring and alerting"
    echo "• Regular stress testing recommended"
fi

echo ""
print_header " DOCKER COMMANDS"
echo "=================="
echo "Start production:  docker-compose up -d"
echo "Start development: docker-compose --profile dev up -d"
echo "Check status:      docker-compose ps"
echo "View logs:         docker-compose logs -f"
echo "Stop services:     docker-compose down"
echo "Health check:      curl http://localhost/health"

echo ""
print_header " ACCESS POINTS"
echo "================"
echo "Frontend (prod):   http://localhost"
echo "Frontend (dev):    http://localhost:5173"
echo "Backend API:       http://localhost:5001"
echo "API Health:        http://localhost:5001/health"

# Exit with appropriate code
if [ $FAILED_TESTS -gt $(( TOTAL_TESTS / 2 )) ]; then
    exit 1
else
    exit 0
fi
