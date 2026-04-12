import requests
import time

API_URL = "http://127.0.0.1:8000"

def test_login_rate_limit():
    print("🛡️  Testing Rate Limiter on /auth/login (Limit: 5 per minute)")
    print("-" * 50)
    
    # The endpoint expects form data for OAuth2
    dummy_payload = {
        "username": "test_spammer@example.com",
        "password": "wrongpassword"
    }

    # We will loop 7 times. The limit is 5, so attempts 6 and 7 MUST be blocked.
    for i in range(1, 8):
        print(f"Attempt {i}/7...", end=" ")
        
        try:
            response = requests.post(f"{API_URL}/auth/login", data=dummy_payload)
            status = response.status_code
            
            if status == 429:
                print(f"🛑 BLOCKED! Status: {status} -> {response.json().get('error', 'Rate limit exceeded')}")
            elif status == 401:
                print(f"✅ Allowed (Failed Login). Status: {status}")
            elif status == 200:
                print(f"✅ Allowed (Success). Status: {status}")
            else:
                print(f"⚠️ Unexpected Status: {status} -> {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Failed. Is the FastAPI server running?")
            break
            
        # Wait half a second between requests
        time.sleep(0.5)

if __name__ == "__main__":
    test_login_rate_limit()