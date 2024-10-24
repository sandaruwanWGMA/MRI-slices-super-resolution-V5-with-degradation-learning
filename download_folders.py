import os
from google.cloud import storage

# Initialize the Google Cloud Storage client
client = storage.Client()

# Define your bucket name
bucket_name = "mri_coupled_dataset"
bucket = client.get_bucket(bucket_name)

# Create a local directory to save the downloaded data
local_directory = "mri_coupled_dataset"
os.makedirs(local_directory, exist_ok=True)

# List and download all files inside the High-Res and Low-Res folders
folders = ["High-Res/", "Low-Res/"]

for folder in folders:
    # List all the blobs in the folder
    blobs = bucket.list_blobs(prefix=folder)

    for blob in blobs:
        # Create local directory for each file based on its path
        local_path = os.path.join(local_directory, blob.name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the file
        print(f"Downloading {blob.name} to {local_path}")
        blob.download_to_filename(local_path)

print("Download of both folders complete!")
