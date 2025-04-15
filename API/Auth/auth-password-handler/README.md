# Password Reset Handler

This AWS Lambda function handles user password reset operations for the Matsight application.

## Features

- Forgot Password: Sends a verification code to the user's email address
- Reset Password: Allows users to set a new password using the verification code
- Multi-environment Support: Dynamically selects configuration based on deployment stage

## API Endpoints

### Forgot Password

Sends a verification code to the user's email for password reset.

**Endpoint**: `/forgot-password`

**Method**: `POST`

**Request Body**:
```json
{
  "email": "user@example.com"
}
```

**Success Response**:
```json
{
  "success": true,
  "data": {
    "message": "Password reset code has been sent to your email",
    "email": "user@example.com"
  }
}
```

**Error Response**:
```json
{
  "success": false,
  "error": {
    "message": "Error message here"
  }
}
```

### Reset Password

Resets the user's password using the verification code received via email.

**Endpoint**: `/reset-password`

**Method**: `POST`

**Request Body**:
```json
{
  "email": "user@example.com",
  "verificationCode": "123456",
  "newPassword": "NewP@ssw0rd"
}
```

**Success Response**:
```json
{
  "success": true,
  "data": {
    "message": "Password has been reset successfully. You can now login with your new password"
  }
}
```

**Error Response**:
```json
{
  "success": false,
  "error": {
    "message": "Error message here"
  }
}
```

## Password Requirements

The new password must:
- Be at least 8 characters long
- Include uppercase letters
- Include lowercase letters
- Include numbers
- Include special characters

## Environment Variables

This Lambda function requires the following environment variables:

### Stage Configuration
- `STAGE`: Deployment environment ('dev' or 'prod')

### Development Environment
- `DEV_REGION`: AWS region for development environment (e.g., 'us-east-1')
- `DEV_APP_CLIENT_ID`: Cognito User Pool Client ID for development
- `DEV_APP_CLIENT_SECRET`: Cognito User Pool Client Secret for development
- `DEV_USER_POOL_ID`: Cognito User Pool ID for development

### Production Environment
- `PROD_REGION`: AWS region for production environment (e.g., 'us-east-1')
- `PROD_APP_CLIENT_ID`: Cognito User Pool Client ID for production
- `PROD_APP_CLIENT_SECRET`: Cognito User Pool Client Secret for production
- `PROD_USER_POOL_ID`: Cognito User Pool ID for production

### Legacy/Fallback Configuration
The following variables are supported for backward compatibility:
- `REGION`: AWS region if stage-specific is not set
- `APP_CLIENT_ID`: Cognito User Pool Client ID if stage-specific is not set
- `APP_CLIENT_SECRET`: Cognito User Pool Client Secret if stage-specific is not set

## Environment Selection

The Lambda function automatically selects the appropriate environment configuration based on the `STAGE` environment variable:
- When `STAGE` is set to 'prod' or 'production', it uses the `PROD_*` variables
- Otherwise, it uses the `DEV_*` variables
- If stage-specific variables are not found, it falls back to the legacy variables

## Deployment

1. Install dependencies:
```
npm install
```

2. Package the Lambda function:
```
zip -r auth-password-handler.zip index.js package.json src/
```

3. Upload to AWS Lambda and configure environment variables according to the deployment stage.

4. Configure API Gateway to route `/forgot-password` and `/reset-password` endpoints to this Lambda function.

5. Set the `STAGE` environment variable appropriately for each deployment. 