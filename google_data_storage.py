from google.cloud import storage

# Initialize the client
client = storage.Client()

bucket_name = "mri_coupled_dataset"
bucket = client.get_bucket(bucket_name)

# List all blobs in a directory and write to a txt file
with open("file_paths.txt", "w") as f:
    blobs = bucket.list_blobs(prefix="High-Res/")  # Adjust your prefix accordingly
    for blob in blobs:
        f.write(f"gs://{bucket_name}/{blob.name}\n")
