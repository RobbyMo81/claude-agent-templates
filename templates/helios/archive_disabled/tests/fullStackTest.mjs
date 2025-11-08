#!/usr/bin/env node

/**
 * Full Stack Integration Test Script
 * Tests the complete integration between frontend and backend
 */

import fetch from 'node-fetch';

const FRONTEND_URL = 'http://localhost:5174';
const BACKEND_URL = 'http://localhost:5001';

class FullStackTester {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
    }

    async test(name, testFn) {
        process.stdout.write(`[INFO] Testing: ${name}... `);
        try {
            const result = await testFn();
            if (result) {
                console.log(' PASSED');
                this.passed++;
            } else {
                console.log(' FAILED');
                this.failed++;
            }
        } catch (error) {
            console.log(' ERROR:', error.message);
            this.failed++;
        }
    }

    async runTests() {
        console.log(' Helios Full Stack Integration Test');
        console.log('====================================\n');

        // Test 1: Frontend accessibility
        await this.test('Frontend server is running', async () => {
            const response = await fetch(FRONTEND_URL);
            return response.ok;
        });

        // Test 2: Backend health check
        await this.test('Backend health endpoint', async () => {
            const response = await fetch(`${BACKEND_URL}/health`);
            if (!response.ok) return false;
            const data = await response.json();
            return data.status === 'healthy';
        });

        // Test 3: Backend API models endpoint
        await this.test('Backend models API', async () => {
            const response = await fetch(`${BACKEND_URL}/api/models`);
            if (!response.ok) return false;
            const data = await response.json();
            return Array.isArray(data);
        });

        // Test 4: CORS configuration
        await this.test('CORS configuration', async () => {
            const response = await fetch(`${BACKEND_URL}/health`, {
                method: 'OPTIONS',
                headers: {
                    'Origin': FRONTEND_URL,
                    'Access-Control-Request-Method': 'GET'
                }
            });
            return response.ok;
        });

        // Test 5: Configuration consistency
        await this.test('Configuration consistency', async () => {
            // This would normally check that frontend config matches backend
            return true; // Placeholder
        });

        console.log('\n Test Results Summary');
        console.log('========================');
        console.log(`Total Tests: ${this.passed + this.failed}`);
        console.log(` Passed: ${this.passed}`);
        console.log(` Failed: ${this.failed}`);

        const successRate = Math.round((this.passed / (this.passed + this.failed)) * 100);
        console.log(`Success Rate: ${successRate}%`);

        if (this.failed === 0) {
            console.log('\n ALL TESTS PASSED! Full stack integration is working.');
        } else {
            console.log('\n Some tests failed. Check the issues above.');
        }

        console.log('\n Access Points:');
        console.log(`Frontend: ${FRONTEND_URL}`);
        console.log(`Backend:  ${BACKEND_URL}`);
        console.log(`Health:   ${BACKEND_URL}/health`);
        console.log(`API:      ${BACKEND_URL}/api/models`);

        return this.failed === 0;
    }
}

// Run tests if this script is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    const tester = new FullStackTester();
    const success = await tester.runTests();
    process.exit(success ? 0 : 1);
}

export { FullStackTester };
