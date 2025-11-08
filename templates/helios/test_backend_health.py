import unittest
import requests

class TestBackendServerHealth(unittest.TestCase):
    BASE_URL = "http://localhost:5001"  # Adjust port if needed

    def test_health_endpoint(self):
        """Test /health endpoint for server status"""
        try:
            response = requests.get(f"{self.BASE_URL}/health", timeout=5)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data.get("status"), "healthy")
            self.assertEqual(data.get("service"), "helios-backend")
        except Exception as e:
            self.fail(f"Health endpoint test failed: {e}")

    def test_api_health_endpoint(self):
        """Test /api/health endpoint for server status"""
        try:
            response = requests.get(f"{self.BASE_URL}/api/health", timeout=5)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data.get("status"), "healthy")
            self.assertEqual(data.get("service"), "helios-backend")
        except Exception as e:
            self.fail(f"API health endpoint test failed: {e}")

if __name__ == "__main__":
    unittest.main()
