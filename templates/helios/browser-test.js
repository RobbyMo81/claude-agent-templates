// Simple backend connectivity test
// Run this in the browser console

console.log('🔍 Diagnosing Full Stack Integration...');
console.log('=====================================');

// Test 1: Check current configuration
import('./services/config.js').then(config => {
    console.log('\n📋 Current Configuration:');
    console.log('Frontend URL:', window.location.origin);
    console.log('Backend URL:', config.getApiBaseUrl());
    console.log('Full Config:', config.getCurrentConfig());

    // Test 2: Simple fetch test
    console.log('\n🔗 Testing Backend Connectivity...');

    fetch(config.getApiEndpoint('/health'))
        .then(response => {
            console.log('Response Status:', response.status);
            console.log('Response OK:', response.ok);

            if (response.ok) {
                return response.json();
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        })
        .then(data => {
            console.log('\n✅ SUCCESS: Backend is responding!');
            console.log('📊 Health Data:', data);

            // Test additional endpoints
            return fetch(config.getApiEndpoint('/api/models'));
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error('Models endpoint failed');
        })
        .then(models => {
            console.log('\n🤖 Models API Test:');
            console.log('Available Models:', models);
            console.log('\n🎉 FULL INTEGRATION SUCCESS!');
        })
        .catch(error => {
            console.log('\n❌ Integration Issue Detected:');
            console.log('Error:', error.message);
            console.log('\n🔧 Troubleshooting Steps:');
            console.log('1. Verify backend is running: npm run python:dev');
            console.log('2. Check backend URL in another tab: http://localhost:5001/health');
            console.log('3. Verify virtual environment is activated');
            console.log('4. Check for any firewall/antivirus blocking connections');
        });
}).catch(err => {
    console.log('❌ Config loading failed:', err);
});
