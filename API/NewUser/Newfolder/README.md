# New User Folder Creation Lambda

This Lambda function creates the required folder structure for new users in the specified S3 bucket.

## Event Format

The Lambda function expects an event with the following JSON structure:

```json
{
    "user_id": "string"  // Required: The unique identifier for the user (e.g., Cognito sub)
}
```

## Folder Structure Created

The function creates the following folder structure in S3:
```
users/
└── {user_id}/
    ├── Original Images/
    ├── Tag Images/
    └── __result/
```

## Response Format

### Success Response (200)
```json
{
    "statusCode": 200,
    "headers": {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*"
    },
    "body": {
        "message": "User folders created successfully",
        "folders": {
            "base_folder": "users/{user_id}/",
            "original_images": "users/{user_id}/Original Images/",
            "tag_images": "users/{user_id}/Tag Images/",
            "result": "users/{user_id}/__result/"
        }
    }
}
```

### Error Responses

#### Bad Request (400)
```json
{
    "statusCode": 400,
    "headers": {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*"
    },
    "body": {
        "error": "Missing required parameter: user_id",
        "message": "Missing required parameter: user_id"
    }
}
```

#### Unauthorized (401)
```json
{
    "statusCode": 401,
    "body": {
        "error": "AWS Error (InvalidToken): The provided token is invalid",
        "message": "AWS Error (InvalidToken): The provided token is invalid"
    }
}
```

#### Forbidden (403)
```json
{
    "statusCode": 403,
    "body": {
        "error": "AWS Error (AccessDenied): Access Denied",
        "message": "AWS Error (AccessDenied): Access Denied"
    }
}
```

#### Not Found (404)
```json
{
    "statusCode": 404,
    "body": {
        "error": "AWS Error (NoSuchBucket): The specified bucket does not exist",
        "message": "AWS Error (NoSuchBucket): The specified bucket does not exist"
    }
}
```

#### Too Many Requests (429)
```json
{
    "statusCode": 429,
    "body": {
        "error": "AWS Error (ThrottlingException): Rate exceeded",
        "message": "AWS Error (ThrottlingException): Rate exceeded"
    }
}
```

#### Internal Server Error (500)
```json
{
    "statusCode": 500,
    "body": {
        "error": "Internal Server Error: [error details]",
        "message": "Internal Server Error: [error details]"
    }
}
```

## Required Environment Variables

- `BUCKET_NAME`: The name of the S3 bucket where user folders will be created

## Required IAM Permissions

The Lambda function requires the following S3 permissions:
- `s3:PutObject`

Example IAM policy:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::{bucket_name}/users/*"
        }
    ]
}
```

## Testing

Example test event:
```json
{
    "user_id": "12345678-1234-1234-1234-123456789012"
} 