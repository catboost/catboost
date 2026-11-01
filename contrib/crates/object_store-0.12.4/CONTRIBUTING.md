<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Development instructions

## Running Tests

Tests can be run using `cargo`

```shell
cargo test
```

## Running Integration Tests

By default, integration tests are not run. To run them you will need to set `TEST_INTEGRATION=1` and then provide the
necessary configuration for that object store

### AWS

To test the S3 integration against [localstack](https://localstack.cloud/)

First start up a container running localstack

```shell
LOCALSTACK_VERSION=sha256:a0b79cb2430f1818de2c66ce89d41bba40f5a1823410f5a7eaf3494b692eed97
podman run -d -p 4566:4566 localstack/localstack@$LOCALSTACK_VERSION
podman run -d -p 1338:1338 amazon/amazon-ec2-metadata-mock:v1.9.2 --imdsv2
```

Setup environment

```shell
export TEST_INTEGRATION=1
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_ENDPOINT=http://localhost:4566
export AWS_ALLOW_HTTP=true
export AWS_BUCKET_NAME=test-bucket
```

Create a bucket using the AWS CLI

```shell
podman run --net=host --env-host amazon/aws-cli --endpoint-url=http://localhost:4566 s3 mb s3://test-bucket
```

Or directly with:

```shell
aws s3 mb s3://test-bucket --endpoint-url=http://localhost:4566
aws --endpoint-url=http://localhost:4566 s3 mb s3://test-bucket-for-spawn
aws --endpoint-url=http://localhost:4566 dynamodb create-table --table-name test-table --key-schema AttributeName=path,KeyType=HASH AttributeName=etag,KeyType=RANGE --attribute-definitions AttributeName=path,AttributeType=S AttributeName=etag,AttributeType=S --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
```

Run tests

```shell
cargo test --features aws
```

#### Encryption tests

To create an encryption key for the tests, you can run the following command:

```shell
export AWS_SSE_KMS_KEY_ID=$(aws --endpoint-url=http://localhost:4566 \
  kms create-key --description "test key" |
  jq -r '.KeyMetadata.KeyId')
```

To run integration tests with encryption, you can set the following environment variables:

```shell
export AWS_SERVER_SIDE_ENCRYPTION=aws:kms
export AWS_SSE_BUCKET_KEY=false
cargo test --features aws
```

As well as:

```shell
unset AWS_SSE_BUCKET_KEY
export AWS_SERVER_SIDE_ENCRYPTION=aws:kms:dsse
cargo test --features aws
```

#### SSE-C Encryption tests

Unfortunately, localstack does not support SSE-C encryption (https://github.com/localstack/localstack/issues/11356).

We will use [MinIO](https://min.io/docs/minio/container/operations/server-side-encryption.html) to test SSE-C encryption.

First, create a self-signed certificate to enable HTTPS for MinIO, as SSE-C requires HTTPS.

```shell
mkdir ~/certs
cd ~/certs
openssl genpkey -algorithm RSA -out private.key
openssl req -new -key private.key -out request.csr -subj "/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=example.com/emailAddress=email@example.com"
openssl x509 -req -days 365 -in request.csr -signkey private.key -out public.crt
rm request.csr
```

Second, start MinIO with the self-signed certificate.

```shell
docker run -d \
  -p 9000:9000 \
  --name minio \
  -v ${HOME}/certs:/root/.minio/certs \
  -e "MINIO_ROOT_USER=minio" \
  -e "MINIO_ROOT_PASSWORD=minio123" \
  minio/minio server /data
```

Create a test bucket.

```shell
export AWS_BUCKET_NAME=test-bucket
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
export AWS_ENDPOINT=https://localhost:9000
aws s3 mb s3://test-bucket --endpoint-url=https://localhost:9000 --no-verify-ssl
```

Run the tests. The real test is `test_s3_ssec_encryption_with_minio()`

```shell
export TEST_S3_SSEC_ENCRYPTION=1
cargo test --features aws --package object_store --lib aws::tests::test_s3_ssec_encryption_with_minio -- --exact --nocapture
```

### Azure

To test the Azure integration
against [azurite](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azurite?tabs=visual-studio)

Startup azurite

```shell
podman run -p 10000:10000 -p 10001:10001 -p 10002:10002 mcr.microsoft.com/azure-storage/azurite
```

Create a bucket

```shell
podman run --net=host mcr.microsoft.com/azure-cli az storage container create -n test-bucket --connection-string 'DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;'
```

Run tests

```shell
AZURE_USE_EMULATOR=1 \
TEST_INTEGRATION=1 \
AZURE_CONTAINER_NAME=test-bucket \
AZURE_STORAGE_ACCOUNT_NAME=devstoreaccount1 \
AZURE_STORAGE_ACCESS_KEY=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw== \
AZURE_ENDPOINT=http://127.0.0.1:10000/devstoreaccount1 \
AZURE_ALLOW_HTTP=true \
cargo test --features azure
```

### GCP

To test the GCS integration, we use [Fake GCS Server](https://github.com/fsouza/fake-gcs-server)

Startup the fake server:

```shell
docker run -p 4443:4443 tustvold/fake-gcs-server -scheme http
```

Configure the account:
```shell
curl -v -X POST --data-binary '{"name":"test-bucket"}' -H "Content-Type: application/json" "http://localhost:4443/storage/v1/b"
echo '{"gcs_base_url": "http://localhost:4443", "disable_oauth": true, "client_email": "", "private_key": ""}' > /tmp/gcs.json
```

Now run the tests:
```shell
TEST_INTEGRATION=1 \
OBJECT_STORE_BUCKET=test-bucket \
GOOGLE_SERVICE_ACCOUNT=/tmp/gcs.json \
cargo test -p object_store --features=gcp
```

# Deprecation Guidelines

Minor releases may deprecate, but not remove APIs. Deprecating APIs allows
downstream Rust programs to still compile, but generate compiler warnings. This
gives downstream crates time to migrate prior to API removal.

To deprecate an API:

- Mark the API as deprecated using `#[deprecated]` and specify the exact object_store version in which it was deprecated
- Concisely describe the preferred API to help the user transition

The deprecated version is the next version which will be released (please
consult the list above). To mark the API as deprecated, use the
`#[deprecated(since = "...", note = "...")]` attribute.

For example

```rust
#[deprecated(since = "0.11.0", note = "Use `date_part` instead")]
```

In general, deprecated APIs will remain in the codebase for at least two major releases after
they were deprecated (typically between 6 - 9 months later). For example, an API
deprecated in `0.10.0` can be removed in `0.13.0` (or later). Deprecated APIs
may be removed earlier or later than these guidelines at the discretion of the
maintainers.
