# Authentication Login Handler

## Quick Summary
- **Purpose**: AWS Lambda function for user authentication
- **Input**: Username (email) and password
- **Output**: accessToken, refreshToken, idToken, userData
- **Technology**: AWS Cognito, Node.js, AWS Lambda
- **Integration**: Part of the authentication API system

## Overview
This Lambda function handles user authentication using AWS Cognito. It processes login requests, validates credentials, and returns authentication tokens along with user information.

## Directory Structure
```
auth-login-handler/
├── index.js                 # Lambda function entry point
├── package.json            # Project dependencies and configuration
└── src/
    ├── services/
    │   └── AuthService.js  # Authentication business logic
    ├── models/
    │   └── User.js        # User data model
    └── utils/
        ├── CognitoClient.js    # AWS Cognito interaction utility
        └── ResponseBuilder.js   # API response formatter
```

## Components Description

### 1. Entry Point (index.js)
- Main Lambda handler function
- Processes incoming requests from both API Gateway and direct Lambda invocations
- Handles request validation and error handling
- Coordinates the authentication flow

### 2. Authentication Service (AuthService.js)
- Implements core authentication logic
- Manages Cognito authentication flow
- Handles user session management
- Processes authentication responses

### 3. User Model (User.js)
- Defines user data structure
- Handles user attribute mapping
- Provides data transformation methods
- Manages user profile information

### 4. Utilities
#### CognitoClient.js
- Manages AWS Cognito service interactions
- Handles token generation and validation
- Implements secure hash calculation
- Manages AWS SDK integration

#### ResponseBuilder.js
- Standardizes API response format
- Handles error response formatting
- Manages HTTP status codes
- Implements CORS headers

## API Specification

### Request Format
```json
{
  "username": "user@example.com",
  "password": "UserPassword123!"
}
```

### Success Response
```json
{
  "success": true,
  "data": {
    "tokens": {
      "accessToken": "eyJhbGciOiJIUzI1...",
      "refreshToken": "eyJhbGciOiJIUzI1...",
      "idToken": "eyJhbGciOiJIUzI1..."
    },
    "user": {
      "username": "John Doe",
      "email": "user@example.com",
      "sub": "user-uuid",
      "lastLogin": "2024-04-11T12:34:56.789Z"
    }
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "message": "Error description"
  }
}
```

## Environment Variables
- `REGION`: AWS region (e.g., "us-east-1")
- `APP_CLIENT_ID`: Cognito App Client ID
- `APP_CLIENT_SECRET`: Cognito App Client Secret

## Error Handling
The function handles various error scenarios:
- Invalid input format
- Missing required fields
- Invalid credentials
- Unverified users
- Cognito service errors
- Network issues

## Security Features
- Secure password handling
- Token-based authentication
- CORS support
- Input validation
- Error message sanitization
- Secure hash calculation

## Testing
To test the function locally or in AWS Lambda console:
1. Create a test event with the request format shown above
2. Execute the test
3. Verify the response format and data

## Deployment
Deploy this function as part of the authentication API:
1. Package the code and dependencies
2. Upload to AWS Lambda
3. Configure environment variables
4. Set up API Gateway integration
5. Configure necessary IAM roles and permissions

## Dependencies
```json
{
  "@aws-sdk/client-cognito-identity-provider": "^3.0.0"
}
```

## Best Practices Implemented
- Object-Oriented Design
- Separation of Concerns
- Error Handling
- Input Validation
- Secure Configuration
- Standardized Responses
- Comprehensive Logging
- SOLID Principles

## Maintenance and Scaling
- Modular architecture for easy updates
- Configurable through environment variables
- Extensible error handling
- Scalable response formatting
- Clear separation of concerns

## Related Components
- Registration Handler
- Password Reset Handler
- Token Refresh Handler
- User Verification Handler 