# Auth Registration Handler

This Lambda function handles user registration, email verification, and resending verification codes for authentication-related functionality.

## Features

1. User Registration (`/register`)
2. Email Verification (`/verify-email`)
3. Resend Verification Code (`/resend-verification`)

## New Feature

When a user successfully registers, this Lambda function automatically calls another Lambda function and passes the newly registered user's Cognito UserSub in the following format:

```json
{
  "user_id": "sub_from_cognito"
}
```

**Important**: The user creation Lambda is only called in the `prod` environment. In all other environments (dev, test, etc.), this call is skipped.

## Target Lambda Function

This function is configured to call the following Lambda ARN in production:
```
arn:aws:lambda:us-east-1:043309364810:function:NewFolder:prod
```

## Configuration

This Lambda function supports the following environment variables:

| Environment Variable | Description | Default Value |
|---------|------|-------|
| `TARGET_LAMBDA_ARN` | Complete ARN of the Lambda function to invoke | `arn:aws:lambda:us-east-1:043309364810:function:NewFolder:prod` |
| `USER_CREATION_LAMBDA` | Name of the user creation Lambda function (when not using ARN) | `user-creation-handler` |
| `LAMBDA_USER_CREATION_HANDLER` | Complete name of the user creation Lambda function (including stage, when not using ARN) | `user-creation-handler-{stage}` |
| `AWS_REGION` | AWS Region | `us-east-1` |
| `STAGE` | Deployment stage | `dev` |

## Stage Detection

The Lambda function automatically detects the stage from:
1. The `STAGE` environment variable
2. The API Gateway stage
3. The request context path
4. If no stage is detected, it defaults to `prod`

## Deployment

1. Install dependencies:
   ```
   npm install
   ```

2. Deploy to AWS Lambda:
   ```
   npm run deploy
   ```

## Permissions

This Lambda function requires the following permissions:

1. Cognito user pool operation permissions
2. Lambda invocation permissions (for calling the target Lambda in production)

Please ensure that the Lambda function role has these permissions. 