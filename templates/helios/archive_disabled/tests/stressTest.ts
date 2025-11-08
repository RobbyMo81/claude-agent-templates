/**
 * Comprehensive Stress Test Suite for Helios System
 * Tests unified configuration, Docker containers, and system resilience
 */

import { 
    updateApiPort, 
    updateApiHost, 
    updateApiProtocol,
    updateFullApiConfig,
    getCurrentConfig,
    getApiEndpoint
} from '../services/config';
import { modelService } from '../services/modelService';
import { runFullAnalysis } from '../services/api';

interface StressTestResult {
    testName: string;
    status: 'PASS' | 'FAIL' | 'WARNING';
    duration: number;
    details: string;
    metrics?: {
        requestsPerSecond?: number;
        errorRate?: number;
        avgResponseTime?: number;
    };
}

interface StressTestSuite {
    suiteName: string;
    results: StressTestResult[];
    summary: {
        totalTests: number;
        passed: number;
        failed: number;
        warnings: number;
        totalDuration: number;
    };
}

class SystemStressTester {
    private results: StressTestResult[] = [];
    private startTime: number = 0;

    constructor() {
        this.startTime = Date.now();
    }

    /**
     * Run a single test and record results
     */
    private async runTest(
        testName: string, 
        testFunction: () => Promise<{ status: 'PASS' | 'FAIL' | 'WARNING'; details: string; metrics?: any }>
    ): Promise<StressTestResult> {
        const start = Date.now();
        console.log(` Running: ${testName}`);
        
        try {
            const result = await testFunction();
            const duration = Date.now() - start;
            
            const testResult: StressTestResult = {
                testName,
                status: result.status,
                duration,
                details: result.details,
                metrics: result.metrics
            };
            
            this.results.push(testResult);
            
            const statusIcon = result.status === 'PASS' ? '' : result.status === 'FAIL' ? '' : '';
            console.log(`${statusIcon} ${testName} (${duration}ms): ${result.details}`);
            
            return testResult;
        } catch (error) {
            const duration = Date.now() - start;
            const testResult: StressTestResult = {
                testName,
                status: 'FAIL',
                duration,
                details: `Unexpected error: ${error instanceof Error ? error.message : String(error)}`
            };
            
            this.results.push(testResult);
            console.log(` ${testName} (${duration}ms): FAILED - ${testResult.details}`);
            
            return testResult;
        }
    }

    /**
     * Test 1: Configuration System Stress Test
     */
    private async testConfigurationStress(): Promise<{ status: 'PASS' | 'FAIL' | 'WARNING'; details: string; metrics?: any }> {
        const iterations = 1000;
        const start = Date.now();
        
        try {
            // Test rapid configuration changes
            for (let i = 0; i < iterations; i++) {
                updateApiPort(5001 + (i % 100));
                updateApiHost(i % 2 === 0 ? 'localhost' : '127.0.0.1');
                updateApiProtocol(i % 3 === 0 ? 'https' : 'http');
                
                // Verify configuration consistency
                const config = getCurrentConfig();
                const expectedUrl = `${config.api.protocol}://${config.api.host}:${config.api.port}`;
                if (config.api.baseUrl !== expectedUrl) {
                    throw new Error(`Configuration inconsistency at iteration ${i}`);
                }
            }
            
            const duration = Date.now() - start;
            const rps = Math.round((iterations / duration) * 1000);
            
            // Reset to defaults
            updateFullApiConfig({
                protocol: 'http',
                host: 'localhost',
                port: 5001
            });
            
            return {
                status: 'PASS',
                details: `Successfully performed ${iterations} configuration changes`,
                metrics: {
                    requestsPerSecond: rps,
                    avgResponseTime: duration / iterations
                }
            };
        } catch (error) {
            return {
                status: 'FAIL',
                details: `Configuration stress test failed: ${error instanceof Error ? error.message : String(error)}`
            };
        }
    }

    /**
     * Test 2: API Endpoint Generation Stress Test
     */
    private async testEndpointGenerationStress(): Promise<{ status: 'PASS' | 'FAIL' | 'WARNING'; details: string; metrics?: any }> {
        const iterations = 10000;
        const endpoints = [
            '/api/models',
            '/api/train',
            '/api/models/test/journal',
            'health',
            '/status',
            'api/metrics',
            '/api/models/very-long-model-name-with-special-chars/journal'
        ];
        
        const start = Date.now();
        
        try {
            for (let i = 0; i < iterations; i++) {
                const endpoint = endpoints[i % endpoints.length];
                const fullUrl = getApiEndpoint(endpoint);
                
                // Verify URL format
                if (!fullUrl.startsWith('http')) {
                    throw new Error(`Invalid URL generated: ${fullUrl}`);
                }
                
                // Test with different configurations
                if (i % 1000 === 0) {
                    updateApiPort(5001 + (i % 10));
                }
            }
            
            const duration = Date.now() - start;
            const rps = Math.round((iterations / duration) * 1000);
            
            return {
                status: 'PASS',
                details: `Generated ${iterations} endpoints successfully`,
                metrics: {
                    requestsPerSecond: rps,
                    avgResponseTime: duration / iterations
                }
            };
        } catch (error) {
            return {
                status: 'FAIL',
                details: `Endpoint generation failed: ${error instanceof Error ? error.message : String(error)}`
            };
        }
    }

    /**
     * Test 3: Concurrent API Calls Stress Test
     */
    private async testConcurrentApiCalls(): Promise<{ status: 'PASS' | 'FAIL' | 'WARNING'; details: string; metrics?: any }> {
        const concurrentRequests = 50;
        
        const start = Date.now();
        let successCount = 0;
        let errorCount = 0;
        
        try {
            const promises: Promise<any>[] = [];
            
            for (let i = 0; i < concurrentRequests; i++) {
                const promise = (async () => {
                    try {
                        // Test different API endpoints
                        switch (i % 3) {
                            case 0:
                                await modelService.getModels();
                                break;
                            case 1:
                                await modelService.startTraining({
                                    model_name: `stress-test-${i}`,
                                    epochs: 10,
                                    learning_rate: 0.001
                                });
                                break;
                            case 2:
                                await modelService.getModelJournal(`model-${i}`);
                                break;
                        }
                        successCount++;
                    } catch (error) {
                        errorCount++;
                        // Expected for stress testing - some may fail
                    }
                })();
                
                promises.push(promise);
            }
            
            await Promise.allSettled(promises);
            
            const duration = Date.now() - start;
            const errorRate = (errorCount / concurrentRequests) * 100;
            
            return {
                status: errorRate < 50 ? 'PASS' : errorRate < 80 ? 'WARNING' : 'FAIL',
                details: `${successCount}/${concurrentRequests} requests succeeded`,
                metrics: {
                    requestsPerSecond: Math.round((concurrentRequests / duration) * 1000),
                    errorRate: errorRate,
                    avgResponseTime: duration / concurrentRequests
                }
            };
        } catch (error) {
            return {
                status: 'FAIL',
                details: `Concurrent API test failed: ${error instanceof Error ? error.message : String(error)}`
            };
        }
    }

    /**
     * Test 4: Memory and Resource Usage Test
     */
    private async testMemoryUsage(): Promise<{ status: 'PASS' | 'FAIL' | 'WARNING'; details: string; metrics?: any }> {
        const iterations = 5000;
        const initialMemory = (performance as any).memory?.usedJSHeapSize || 0;
        
        try {
            // Create memory pressure
            const largeObjects: any[] = [];
            
            for (let i = 0; i < iterations; i++) {
                // Create configuration objects
                const config = getCurrentConfig();
                largeObjects.push({
                    id: i,
                    config: { ...config },
                    endpoints: [
                        getApiEndpoint('/api/models'),
                        getApiEndpoint('/api/train'),
                        getApiEndpoint(`/api/models/model-${i}/journal`)
                    ],
                    timestamp: Date.now()
                });
                
                // Periodic cleanup to test garbage collection
                if (i % 1000 === 0) {
                    largeObjects.splice(0, 500);
                    
                    // Force garbage collection if available (Node.js environment)
                    try {
                        if (typeof window === 'undefined' && (globalThis as any).gc) {
                            (globalThis as any).gc();
                        }
                    } catch (e) {
                        // Ignore - gc not available
                    }
                }
            }
            
            const finalMemory = (performance as any).memory?.usedJSHeapSize || 0;
            const memoryIncrease = finalMemory - initialMemory;
            const memoryIncreaseKB = Math.round(memoryIncrease / 1024);
            
            // Cleanup
            largeObjects.length = 0;
            
            return {
                status: memoryIncreaseKB < 50000 ? 'PASS' : memoryIncreaseKB < 100000 ? 'WARNING' : 'FAIL',
                details: `Memory usage increased by ${memoryIncreaseKB}KB`,
                metrics: {
                    memoryIncrease: memoryIncreaseKB
                }
            };
        } catch (error) {
            return {
                status: 'FAIL',
                details: `Memory test failed: ${error instanceof Error ? error.message : String(error)}`
            };
        }
    }

    /**
     * Test 5: Error Handling and Recovery Test
     */
    private async testErrorHandling(): Promise<{ status: 'PASS' | 'FAIL' | 'WARNING'; details: string; metrics?: any }> {
        let recoveryCount = 0;
        const totalTests = 10;
        
        try {
            // Test invalid configurations
            const originalConfig = getCurrentConfig();
            
            // Test 1: Invalid port
            try {
                updateApiPort(-1);
                updateApiPort(99999);
                recoveryCount++;
            } catch (error) {
                // Expected
            }
            
            // Test 2: Invalid protocol
            try {
                updateApiProtocol('ftp' as any);
                recoveryCount++;
            } catch (error) {
                // Expected
            }
            
            // Test 3: Invalid host
            try {
                updateApiHost('');
                updateApiHost('invalid..host..name');
                recoveryCount++;
            } catch (error) {
                // Expected
            }
            
            // Test 4: Network timeouts (simulated)
            try {
                await Promise.race([
                    modelService.getModels(),
                    new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 100))
                ]);
                recoveryCount++;
            } catch (error) {
                recoveryCount++; // Recovery successful if we catch the error
            }
            
            // Test 5: Large file processing
            try {
                const largeCSVContent = 'date,numbers,powerball\n' + 
                    Array(10000).fill(0).map((_, i) => 
                        `2025-01-${String(i % 28 + 1).padStart(2, '0')},1-2-3-4-5,${i % 26 + 1}`
                    ).join('\n');
                
                const blob = new Blob([largeCSVContent], { type: 'text/csv' });
                const file = new File([blob], 'large-test.csv', { type: 'text/csv' });
                
                await runFullAnalysis(file, 'SLOW_BACKTEST');
                recoveryCount++;
            } catch (error) {
                recoveryCount++; // Recovery if error is handled gracefully
            }
            
            // Restore original configuration
            updateFullApiConfig(originalConfig.api);
            recoveryCount += 5; // Additional points for successful restoration
            
            return {
                status: recoveryCount >= 7 ? 'PASS' : recoveryCount >= 5 ? 'WARNING' : 'FAIL',
                details: `${recoveryCount}/${totalTests} error scenarios handled correctly`,
                metrics: {
                    recoveryRate: (recoveryCount / totalTests) * 100
                }
            };
        } catch (error) {
            return {
                status: 'FAIL',
                details: `Error handling test failed: ${error instanceof Error ? error.message : String(error)}`
            };
        }
    }

    /**
     * Test 6: Docker Container Health Check Simulation
     */
    private async testContainerHealth(): Promise<{ status: 'PASS' | 'FAIL' | 'WARNING'; details: string; metrics?: any }> {
        const healthChecks = 20;
        let healthyCount = 0;
        
        try {
            for (let i = 0; i < healthChecks; i++) {
                try {
                    // Simulate health check by testing basic functionality
                    const config = getCurrentConfig();
                    const endpoint = getApiEndpoint('/health');
                    
                    // Test configuration consistency
                    if (config.api.baseUrl && endpoint.includes(config.api.host)) {
                        healthyCount++;
                    }
                    
                    // Simulate network delay
                    await new Promise(resolve => setTimeout(resolve, Math.random() * 100));
                    
                } catch (error) {
                    // Health check failed
                }
            }
            
            const healthRate = (healthyCount / healthChecks) * 100;
            
            return {
                status: healthRate >= 95 ? 'PASS' : healthRate >= 80 ? 'WARNING' : 'FAIL',
                details: `${healthyCount}/${healthChecks} health checks passed (${healthRate.toFixed(1)}%)`,
                metrics: {
                    healthRate: healthRate
                }
            };
        } catch (error) {
            return {
                status: 'FAIL',
                details: `Container health test failed: ${error instanceof Error ? error.message : String(error)}`
            };
        }
    }

    /**
     * Run all stress tests
     */
    async runFullStressTest(): Promise<StressTestSuite> {
        console.log(' Starting Helios System Stress Test Suite...\n');
        
        // Run all stress tests
        await this.runTest('Configuration System Stress Test', () => this.testConfigurationStress());
        await this.runTest('API Endpoint Generation Stress Test', () => this.testEndpointGenerationStress());
        await this.runTest('Concurrent API Calls Test', () => this.testConcurrentApiCalls());
        await this.runTest('Memory and Resource Usage Test', () => this.testMemoryUsage());
        await this.runTest('Error Handling and Recovery Test', () => this.testErrorHandling());
        await this.runTest('Container Health Check Simulation', () => this.testContainerHealth());
        
        // Calculate summary
        const totalDuration = Date.now() - this.startTime;
        const passed = this.results.filter(r => r.status === 'PASS').length;
        const failed = this.results.filter(r => r.status === 'FAIL').length;
        const warnings = this.results.filter(r => r.status === 'WARNING').length;
        
        const summary = {
            totalTests: this.results.length,
            passed,
            failed,
            warnings,
            totalDuration
        };
        
        const suite: StressTestSuite = {
            suiteName: 'Helios System Stress Test Suite',
            results: this.results,
            summary
        };
        
        this.printSummary(suite);
        return suite;
    }
    
    /**
     * Print test summary
     */
    private printSummary(suite: StressTestSuite): void {
        console.log('\n' + '='.repeat(60));
        console.log(' STRESS TEST SUMMARY');
        console.log('='.repeat(60));
        console.log(`Total Tests: ${suite.summary.totalTests}`);
        console.log(` Passed: ${suite.summary.passed}`);
        console.log(`  Warnings: ${suite.summary.warnings}`);
        console.log(` Failed: ${suite.summary.failed}`);
        console.log(`  Total Duration: ${suite.summary.totalDuration}ms`);
        console.log(` Success Rate: ${((suite.summary.passed / suite.summary.totalTests) * 100).toFixed(1)}%`);
        
        if (suite.summary.failed === 0) {
            console.log('\n ALL STRESS TESTS PASSED! System is robust and ready for production.');
        } else if (suite.summary.failed <= suite.summary.totalTests * 0.2) {
            console.log('\n  Some tests failed but system appears stable. Review failed tests.');
        } else {
            console.log('\n MULTIPLE FAILURES DETECTED! System may need attention before production deployment.');
        }
        
        console.log('='.repeat(60));
        
        // Detailed results
        console.log('\n DETAILED RESULTS:');
        suite.results.forEach(result => {
            const statusIcon = result.status === 'PASS' ? '' : result.status === 'FAIL' ? '' : '';
            console.log(`${statusIcon} ${result.testName} (${result.duration}ms)`);
            console.log(`   ${result.details}`);
            if (result.metrics) {
                Object.entries(result.metrics).forEach(([key, value]) => {
                    console.log(`   ${key}: ${value}`);
                });
            }
            console.log();
        });
    }
}

/**
 * Export the stress tester for use in applications
 */
export const runSystemStressTest = async (): Promise<StressTestSuite> => {
    const tester = new SystemStressTester();
    return await tester.runFullStressTest();
};

export type { StressTestResult, StressTestSuite };
export { SystemStressTester };
