import boto3
import shutil
import os
import argparse
import zipfile

def download(path: str, bucket: str, dir: bool = False):
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin"
    )
    s3.download_file(bucket, path, path)
    if dir:
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(path)
        os.remove(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the directory to download")
    parser.add_argument("bucket", type=str, help="Name of the S3 bucket")
    parser.add_argument("--dir", action="store_true", help="Download a directory as a zip file")
    args = parser.parse_args()
    download(args.path, args.bucket, args.dir)