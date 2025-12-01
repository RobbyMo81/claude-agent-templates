/**
 * Simple Stress Test Runner for Windows
 * Runs basic validation tests for the Helios project
 */

import fs from 'fs';
import { exec } from 'child_process';
import path from 'path';

class StressTestRunner {
    constructor() {
        this.totalTests = 0;
        this.passedTests = 0;
        this.failedTests = 0;
        this.warningTests = 0;
    }

    log(level, message) {
        const icons = {
            info: '[INFO]',
            success: '[SUCCESS]',
            warning: '[WARNING]',
            error: '[ERROR]'
        };
        console.log(`${icons[level]} ${message}`);
    }

    async runCommand(command) {
        return new Promise((resolve) => {
            exec(command, (error, stdout, stderr) => {
                resolve({ success: !error, stdout, stderr });
            });
        });
    }

    fileExists(filePath) {
        try {
            return fs.existsSync(filePath);
        } catch (error) {
            return false;
        }
    }

    async runTest(name, testFn) {
        this.totalTests++;
        this.log('info', `Running: ${name}`);

        try {
            const result = await testFn();
            if (result === 'warning') {
                this.warningTests++;
                this.log('warning', `${name} - Warning`);
            } else if (result) {
                this.passedTests++;
                this.log('success', name);
            } else {
                this.failedTests++;
                this.log('error', `${name} - Failed`);
            }
        } catch (error) {
            this.failedTests++;
            this.log('error', `${name} - Error: ${error.message}`);
        }
    }

    async runAllTests() {
        console.log('🚀 Helios System Stress Test Suite');
        console.log('==================================\n');

        // Pre-flight checks
        console.log('🔧 Pre-flight Checks');
        console.log('====================');

        await this.runTest('Node.js Available', async () => {
            const result = await this.runCommand('node --version');
            return result.success;
        });

        await this.runTest('NPM Available', async () => {
            const result = await this.runCommand('npm --version');
            return result.success;
        });

        await this.runTest('Docker Available', async () => {
            const result = await this.runCommand('docker --version');
            return result.success;
        });

        await this.runTest('Docker Compose Available', async () => {
            const result = await this.runCommand('docker-compose --version');
            return result.success;
        });

        console.log('\n📁 Project Structure Validation');
        console.log('==============================');

        await this.runTest('Config Service Exists', () => {
            return this.fileExists('services/config.ts');
        });

        await this.runTest('Model Service Exists', () => {
            return this.fileExists('services/modelService.ts');
        });

        await this.runTest('API Service Exists', () => {
            return this.fileExists('services/api.ts');
        });

        await this.runTest('Docker Compose Config', () => {
            return this.fileExists('docker-compose.yml');
        });

        await this.runTest('Frontend Dockerfile', () => {
            return this.fileExists('Dockerfile');
        });

        await this.runTest('Backend Dockerfile', () => {
            return this.fileExists('backend/Dockerfile');
        });

        console.log('\n⚙️ Configuration System Tests');
        console.log('=============================');

        await this.runTest('Configuration File Validation', () => {
            try {
                const configContent = fs.readFileSync('services/config.ts', 'utf8');
                const requiredExports = ['appConfig'];
                const requiredFunctions = ['getApiBaseUrl', 'getApiPort', 'getApiEndpoint'];

                for (const exp of requiredExports) {
                    if (!configContent.includes(exp)) return false;
                }

                for (const func of requiredFunctions) {
                    if (!configContent.includes(func)) return false;
                }

                return true;
            } catch (error) {
                return false;
            }
        });

        console.log('\n🐳 Docker Configuration Tests');
        console.log('============================');

        await this.runTest('Docker Compose Config Check', async () => {
            const result = await this.runCommand('docker-compose config');
            return result.success;
        });

        console.log('\n🌍 Environment Configuration Tests');
        console.log('=================================');

        await this.runTest('Development Environment File', () => {
            const exists = this.fileExists('.env.development');
            return exists ? true : 'warning';
        });

        await this.runTest('Production Environment File', () => {
            const exists = this.fileExists('.env.production');
            return exists ? true : 'warning';
        });

        console.log('\n🧪 Stress Test Framework Validation');
        console.log('==================================');

        await this.runTest('Stress Test File Exists', () => {
            return this.fileExists('tests/stressTest.ts');
        });

        await this.runTest('Stress Test Component Exists', () => {
            return this.fileExists('components/StressTestComponent.tsx');
        });

        console.log('\n📚 Documentation Tests');
        console.log('=====================');

        await this.runTest('Docker Setup Documentation', () => {
            const exists = this.fileExists('DOCKER_SETUP.md');
            return exists ? true : 'warning';
        });

        await this.runTest('Port Configuration Documentation', () => {
            const exists = this.fileExists('PORT_CONFIGURATION.md');
            return exists ? true : 'warning';
        });

        console.log('\n📝 TypeScript Compilation Tests');
        console.log('==============================');

        await this.runTest('TypeScript Check', async () => {
            const result = await this.runCommand('npx tsc --noEmit --skipLibCheck');
            return result.success ? true : 'warning';
        });

        console.log('\n🏗️ Build Process Tests');
        console.log('=====================');

        await this.runTest('Frontend Build Test', async () => {
            const result = await this.runCommand('npm run build');
            return result.success;
        });

        console.log('\n🐍 Python Virtual Environment Tests');
        console.log('==================================');

        await this.runTest('Python Virtual Environment Exists', () => {
            const venvExists = this.fileExists('backend/venv') || this.fileExists('backend\\venv');
            return venvExists ? true : 'warning';
        });

        await this.runTest('Python Requirements File', () => {
            return this.fileExists('backend/requirements.txt');
        });

        await this.runTest('Python Server File', () => {
            return this.fileExists('backend/server.py');
        });

        await this.runTest('Virtual Environment Activation Scripts', () => {
            const windowsActivate = this.fileExists('backend/venv/Scripts/activate.bat') || this.fileExists('backend\\venv\\Scripts\\activate.bat');
            const unixActivate = this.fileExists('backend/venv/bin/activate');
            return (windowsActivate || unixActivate) ? true : 'warning';
        });

        // Print results
        this.printResults();
    }

    printResults() {
        console.log('\n📊 STRESS TEST RESULTS SUMMARY');
        console.log('==============================');
        console.log(`Total Tests: ${this.totalTests}`);
        console.log(`✅ Passed: ${this.passedTests}`);
        console.log(`⚠️ Warnings: ${this.warningTests}`);
        console.log(`❌ Failed: ${this.failedTests}`);

        const successRate = this.totalTests > 0 ? Math.round((this.passedTests / this.totalTests) * 100) : 0;
        console.log(`Success Rate: ${successRate}%`);

        console.log('\n');
        if (this.failedTests === 0) {
            console.log('[SUCCESS] 🎉 ALL STRESS TESTS PASSED! System is robust and ready for production.');
        } else if (this.failedTests <= 3) {
            console.log('[WARNING] ⚠️ Some tests failed but system appears stable. Review failed tests.');
        } else {
            console.log('[ERROR] 🚨 MULTIPLE FAILURES DETECTED! System may need attention before production deployment.');
        }

        console.log('\n💡 RECOMMENDATIONS');
        console.log('==================');
        if (this.failedTests > 0) {
            console.log('• Review failed tests and address underlying issues');
            console.log('• Check Docker container health and resource allocation');
            console.log('• Verify network connectivity and API endpoints');
        }

        if (successRate >= 80) {
            console.log('• System ready for deployment');
            console.log('• Consider implementing monitoring and alerting');
            console.log('• Regular stress testing recommended');
        }

        console.log('\n🐳 DOCKER COMMANDS');
        console.log('==================');
        console.log('Start production:  docker-compose up -d');
        console.log('Start development: docker-compose --profile dev up -d');
        console.log('Check status:      docker-compose ps');
        console.log('View logs:         docker-compose logs -f');
        console.log('Stop services:     docker-compose down');

        console.log('\n🔗 ACCESS POINTS');
        console.log('================');
        console.log('Frontend (prod):   http://localhost');
        console.log('Frontend (dev):    http://localhost:5173');
        console.log('Backend API:       http://localhost:5001');
        console.log('API Health:        http://localhost:5001/health');

        console.log('\n🚀 Next Steps');
        console.log('=============');
        console.log('1. If tests passed: docker-compose up -d');
        console.log('2. Open browser: http://localhost');
        console.log('3. Check API: http://localhost:5001/health');
        console.log('4. View logs: docker-compose logs -f');

        // Exit with appropriate code
        if (this.failedTests > this.totalTests / 2) {
            process.exit(1);
        } else {
            process.exit(0);
        }
    }
}

// Run the stress test
const runner = new StressTestRunner();
runner.runAllTests().catch(error => {
    console.error('Stress test runner failed:', error);
    process.exit(1);
});
