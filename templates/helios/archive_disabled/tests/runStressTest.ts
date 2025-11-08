/**
 * Stress Test Runner for Helios System
 * This file can be executed to run comprehensive stress tests
 */

import { runSystemStressTest } from './stressTest';

// Add console styling for better output
const colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m'
};

function colorLog(message: string, color: string = colors.reset) {
    console.log(`${color}${message}${colors.reset}`);
}

async function main() {
    try {
        colorLog(' Helios System Stress Test Runner', colors.cyan + colors.bright);
        colorLog('='.repeat(50), colors.blue);
        
        // Display system information
        colorLog('\n System Information:', colors.yellow);
        colorLog(`Platform: ${typeof navigator !== 'undefined' ? navigator.platform : 'Node.js'}`);
        colorLog(`User Agent: ${typeof navigator !== 'undefined' ? navigator.userAgent : 'Server Environment'}`);
        colorLog(`Memory: ${typeof performance !== 'undefined' && (performance as any).memory ? 
            `${Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024)}MB used` : 'N/A'}`);
        colorLog(`Timestamp: ${new Date().toISOString()}`);
        
        colorLog('\n Pre-flight Checks:', colors.yellow);
        
        // Check if all required modules are available
        try {
            const { getCurrentConfig } = await import('../services/config');
            const config = getCurrentConfig();
            colorLog(` Configuration service loaded - API: ${config.api.baseUrl}`, colors.green);
        } catch (error) {
            colorLog(` Configuration service failed: ${error}`, colors.red);
            return;
        }
        
        try {
            await import('../services/modelService');
            colorLog(' Model service loaded', colors.green);
        } catch (error) {
            colorLog(` Model service failed: ${error}`, colors.red);
            return;
        }
        
        try {
            await import('../services/api');
            colorLog(' API service loaded', colors.green);
        } catch (error) {
            colorLog(` API service failed: ${error}`, colors.red);
            return;
        }
        
        colorLog('\n Running Stress Tests...', colors.magenta + colors.bright);
        colorLog('This may take several minutes...', colors.yellow);
        
        // Run the comprehensive stress test
        const startTime = Date.now();
        const results = await runSystemStressTest();
        const endTime = Date.now();
        
        // Additional summary
        colorLog('\n Final Assessment:', colors.cyan + colors.bright);
        colorLog(`Total Execution Time: ${(endTime - startTime) / 1000}s`, colors.blue);
        
        const passRate = (results.summary.passed / results.summary.totalTests) * 100;
        
        if (passRate === 100) {
            colorLog(' EXCELLENT: All tests passed! System is production-ready.', colors.green + colors.bright);
        } else if (passRate >= 80) {
            colorLog(' GOOD: Most tests passed. System is stable with minor issues.', colors.green);
        } else if (passRate >= 60) {
            colorLog('  ACCEPTABLE: System functional but needs attention.', colors.yellow);
        } else {
            colorLog(' CRITICAL: Multiple failures detected. System needs fixes.', colors.red + colors.bright);
        }
        
        // Generate recommendations
        colorLog('\n Recommendations:', colors.cyan);
        
        if (results.summary.failed > 0) {
            colorLog('• Review failed tests and address underlying issues', colors.yellow);
            colorLog('• Check Docker container health and resource allocation', colors.yellow);
            colorLog('• Verify network connectivity and API endpoints', colors.yellow);
        }
        
        if (results.summary.warnings > 0) {
            colorLog('• Monitor warning conditions in production', colors.yellow);
            colorLog('• Consider optimizing resource usage and performance', colors.yellow);
        }
        
        if (passRate >= 80) {
            colorLog('• System ready for deployment', colors.green);
            colorLog('• Consider implementing monitoring and alerting', colors.green);
            colorLog('• Regular stress testing recommended', colors.green);
        }
        
        // Docker-specific recommendations
        colorLog('\n Docker Deployment Checklist:', colors.blue);
        colorLog('• Verify Docker containers are running: docker-compose ps', colors.blue);
        colorLog('• Check container health: docker-compose exec frontend wget -q -O- http://localhost/', colors.blue);
        colorLog('• Monitor resource usage: docker stats', colors.blue);
        colorLog('• Review logs: docker-compose logs -f', colors.blue);
        
        colorLog('\n Test Results Summary:', colors.magenta);
        results.results.forEach(result => {
            const icon = result.status === 'PASS' ? '' : result.status === 'FAIL' ? '' : '';
            const color = result.status === 'PASS' ? colors.green : result.status === 'FAIL' ? colors.red : colors.yellow;
            colorLog(`${icon} ${result.testName}: ${result.details}`, color);
        });
        
        colorLog('\n Next Steps:', colors.cyan);
        colorLog('• If running in Docker: docker-compose logs -f', colors.blue);
        colorLog('• If running locally: npm run dev', colors.blue);
        colorLog('• Access frontend: http://localhost (production) or http://localhost:5173 (development)', colors.blue);
        colorLog('• Access backend API: http://localhost:5001', colors.blue);
        
        return results;
        
    } catch (error) {
        colorLog('\n Stress Test Runner Failed!', colors.red + colors.bright);
        colorLog(`Error: ${error instanceof Error ? error.message : String(error)}`, colors.red);
        
        if (error instanceof Error && error.stack) {
            colorLog('\nStack trace:', colors.red);
            colorLog(error.stack, colors.red);
        }
        
        colorLog('\n Troubleshooting:', colors.yellow);
        colorLog('• Ensure all dependencies are installed: npm install', colors.yellow);
        colorLog('• Check that services are running: docker-compose ps', colors.yellow);
        colorLog('• Verify configuration in services/config.ts', colors.yellow);
        
        throw error;
    }
}

export { main as runStressTestRunner };
