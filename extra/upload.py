import boto3
import shutil
from pathlib import Path
import argparse
import os

def upload(path: str, bucket: str, dir: bool = False):
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin"
    )

    if dir:
        zip_name = path.split("/")[-1]
        shutil.make_archive(zip_name, "zip", path)
        print(f"Compressed directory {path} to {zip_name}.zip")
        zip_name += ".zip"
        s3.upload_file(zip_name, bucket, zip_name)
        print(f"Uploaded compressed directory {zip_name} to {bucket}")
        os.remove(zip_name)
    else:
        s3.upload_file(path, bucket, Path(path).name)
        print(f"Uploaded file {Path(path).name} to {bucket}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the directory to upload")
    parser.add_argument("bucket", type=str, help="Name of the S3 bucket")
    parser.add_argument("--dir", action="store_true", help="Upload a directory as a zip file")
    args = parser.parse_args()
    upload(args.path, args.bucket, args.dir)