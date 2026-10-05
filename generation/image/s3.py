from __future__ import annotations

import os
from typing import Optional

import boto3


def upload_bytes_to_s3(data: bytes, bucket: str, key: str, content_type: str = "image/png") -> str:
    """Upload raw bytes to S3 and return s3:// URL. Requires AWS credentials in env or IAM role."""
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return f"s3://{bucket}/{key}"


def upload_file_to_s3(path: str, bucket: str, key: Optional[str] = None) -> str:
    key = key or path.replace("\\", "/").lstrip("/")
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    s3.upload_file(path, bucket, key)
    return f"s3://{bucket}/{key}"
