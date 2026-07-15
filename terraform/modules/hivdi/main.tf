# Data sources
data "aws_caller_identity" "current" {}

#data "aws_cloudwatch_log_group" "cw_log_group" {
#  name = "/aws/batch/job/${var.prefix}-hivdi/"
#}

data "aws_efs_file_system" "aws_efs_input" {
  creation_token = "${var.prefix}-input"
}

data "aws_efs_file_system" "aws_efs_flpe" {
  creation_token = "${var.prefix}-flpe"
}

data "aws_iam_role" "job_role" {
  count = var.iam_job_role_arn == null ? 1 : 0
  name  = "${var.prefix}-batch-job-role"
}

data "aws_iam_role" "exe_role" {
  count = var.iam_execution_role_arn == null ? 1 : 0
  name  = "${var.prefix}-ecs-exe-task-role"
}

# Local variables
locals {
  account_id             = data.aws_caller_identity.current.account_id
  iam_job_role_arn       = var.iam_job_role_arn != null ? var.iam_job_role_arn : data.aws_iam_role.job_role[0].arn
  iam_execution_role_arn = var.iam_execution_role_arn != null ? var.iam_execution_role_arn : data.aws_iam_role.exe_role[0].arn
  default_tags = length(var.default_tags) == 0 ? {
    application : var.app_name,
    environment : var.environment,
    version : var.app_version
  } : var.default_tags
}
