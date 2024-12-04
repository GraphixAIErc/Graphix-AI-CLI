import os
import requests
import json


def upload_file(server_url, file_path):
    """Upload a file to server URL."""
    try:
        # Open the file in binary mode
        with open(file_path, 'rb') as f:
            # Create a dictionary to hold the file
            files = {
                'file': f,
            }

            # Send a POST request to the server
            response = requests.post(server_url, files=files)

        # Check the response status
        if response.status_code == 200:
            print("File uploaded successfully!")
            response_dict = json.loads(response.text)
            return response_dict
        else:
            print(f"Failed to upload file. Status code: {
                  response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"An error occurred: {e}")


def download_file(url, local_path):
    """Download a file from a URL to a local path if not exists."""
    if not os.path.exists(local_path):
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request succeeds
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {local_path}")
    else:
        print(f"Using cached version: {local_path}")


def setup_local_cache(urls, cache_dir='./cache'):
    """Ensure all files are downloaded and cached."""
    os.makedirs(cache_dir, exist_ok=True)
    local_paths = {}
    for key, url in urls.items():
        # Assumes filename is the last segment of URL
        local_file = os.path.join(cache_dir, url.split('/')[-1])
        download_file(url, local_file)
        local_paths[key] = local_file
    return local_paths
