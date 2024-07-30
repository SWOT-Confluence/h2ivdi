#!/bin/bash
#
# Script to deploy Terraform and Docker image AWS infrastructure
#
# REQUIRES:
#   jq (https://jqlang.github.io/jq/)
#   docker (https://docs.docker.com/desktop/) > version Docker 1.5
#   AWS CLI (https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
#   Terraform (https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
#
# Command line arguments:
# [1] hivdi_image_path: Path to HIVDI container image (tar.gz)
# [1] registry: Registry URI
# [2] repository: Name of repository to create
# [3] prefix: Prefix to use for AWS resources associated with environment deploying to
# [4] s3_state_bucket: Name of the S3 bucket to store Terraform state in (no need for s3:// prefix)
# [5] profile: Name of profile used to authenticate AWS CLI commands
# 
# Example usage: ./deploy.sh "path-to-hivdi-container-image" "account-id.dkr.ecr.region.amazonaws.com" "container-image-name" "prefix-for-environment" "s3-state-bucket-name" "confluence-named-profile" 

HIVDI_IMAGE=$1
REGISTRY=$2
REPOSITORY=$3
PREFIX=$4
S3_STATE=$5
PROFILE=$6

# Load Docker image
docker load < $HIVDI_IMAGE

# Deploy Container Image
./deploy-ecr.sh $REGISTRY $REPOSITORY $PREFIX $PROFILE

# Deploy Terraform
cd terraform/
terraform init -reconfigure -backend-config="bucket=$S3_STATE" -backend-config="key=hivdi.tfstate" -backend-config="region=us-west-2" -backend-config="profile=$PROFILE"
terraform apply -var-file="conf.tfvars" -auto-approve
cd ..