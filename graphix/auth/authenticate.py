import requests
from graphix.config.settings import SERVER_URL, CLIENT_URL
import webbrowser
import time
import keyring

def login():
    """Handles user login with server validation."""
    
    # Request session ID from server
    print("Requesting session ID from the server...")
    response = requests.post(f'{SERVER_URL}/api/generate/session-id', json={'deviceID': 'macOS'})
    session_id = response.json().get('sessionId')

    # Open browser for user login and confirmation
    print("Opening browser for user login and confirmation...")
    print(f'URL: {CLIENT_URL}/verify?sessionId={session_id}')
    webbrowser.open(f'{CLIENT_URL}/verify?sessionId={session_id}')

    # Poll the server for the CLI token
    while True:
        token_response = requests.post(f'{SERVER_URL}/api/check/session-id/{session_id}')
        if token_response.status_code == 200:
            token = token_response.json().get('token')
            publicAddress = token_response.json().get('publicAddress')
            keyring.set_password("system", "graphix_jwt_token",token)
            print(f"Login successful. Public Address: {publicAddress}")

            break
        time.sleep(5)  # Wait before polling again