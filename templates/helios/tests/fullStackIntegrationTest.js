/**
 * Full Stack Integration Test
 * Tests the connection between frontend and backend
 */

import { getApiBaseUrl, getApiEndpoint, appConfig } from '../services/config.js';

async function testFullStackIntegration() {
    console.log('🔗 Testing Full Stack Integration');
    console.log('================================');

    // Test 1: Configuration
    console.log('\n1. Configuration Test:');
    console.log(`   Frontend port: ${window.location.port}`);
    console.log(`   Backend URL: ${getApiBaseUrl()}`);
    console.log(`   API Health endpoint: ${getApiEndpoint('/health')}`);

    // Test 2: Backend connectivity
    console.log('\n2. Backend Connectivity Test:');
    try {
        const response = await fetch(getApiEndpoint('/health'));
        if (response.ok) {
            const data = await response.text();
            console.log('   ✅ Backend is accessible');
            console.log(`   Response: ${data}`);
        } else {
            console.log(`   ⚠️ Backend responded with status: ${response.status}`);
        }
    } catch (error) {
        console.log('   ❌ Backend not accessible');
        console.log(`   Error: ${error.message}`);
        console.log('   💡 Make sure the Python backend is running');
    }

    // Test 3: CORS Test
    console.log('\n3. CORS Configuration Test:');
    try {
        const response = await fetch(getApiEndpoint('/health'), {
            method: 'OPTIONS'
        });
        console.log(`   CORS preflight status: ${response.status}`);
    } catch (error) {
        console.log(`   CORS test failed: ${error.message}`);
    }

    // Test 4: Configuration consistency
    console.log('\n4. Configuration Consistency:');
    console.log(`   Current config: ${JSON.stringify(appConfig, null, 2)}`);

    console.log('\n🎯 Integration Test Complete');
    console.log('============================');
}

// Run the test automatically when loaded
if (typeof window !== 'undefined') {
    // Run test when page loads
    window.addEventListener('load', testFullStackIntegration);

    // Also make it available globally for manual testing
    window.testFullStackIntegration = testFullStackIntegration;

    console.log('💡 Full stack integration test loaded.');
    console.log('   Run testFullStackIntegration() in the browser console to test manually.');
}

export { testFullStackIntegration };
