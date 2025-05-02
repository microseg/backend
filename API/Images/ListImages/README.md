# List Images Lambda Function

This Lambda function is designed to list images stored in an S3 bucket, excluding the `__results` folder for each user.

## Input Event Format

The Lambda function accepts a JSON event with the following structure:

```json
{
    "prefix": "folder/"  // Optional parameter to list images in a specific folder
}
```

### Parameters

- `prefix` (optional): A string representing the folder path within the users' directory to list images from. 
  - If not provided, lists all images from the root of the users' directory
  - If provided, lists images only from the specified folder
  - Example: `"my-folder/"` will list images from `users/my-folder/`

## Environment Variables

The function requires the following environment variable to be set:

- `BUCKET_NAME`: The name of the S3 bucket where images are stored

## Response Format

The function returns a JSON response with the following structure:

### Success Response (200)
```json
{
    "statusCode": 200,
    "headers": {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*"
    },
    "body": {
        "images": [
            {
                "key": "users/example/image.jpg",
                "name": "image.jpg",
                "size": 1234567,
                "last_modified": "2024-01-01T00:00:00.000Z",
                "etag": "\"abc123\"",
                "type": "file"
            }
        ],
        "folders": [
            {
                "key": "users/example/subfolder/",
                "name": "subfolder",
                "type": "folder"
            }
        ],
        "count": 2
    }
}
```

### Error Response (500)
```json
{
    "statusCode": 500,
    "headers": {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*"
    },
    "body": {
        "error": "Error message description"
    }
}
```

## Features

- Lists all images and folders in the specified S3 bucket path
- Automatically excludes the `__results` folder from both files and folders
- Separates files and folders into distinct lists for easier frontend handling
- Provides detailed information about each file (key, name, size, last modified date, etag)
- Provides simplified information for folders (key, name)
- Uses S3 delimiter to properly handle folder structures
- Includes CORS headers for cross-origin access
- Handles various error cases gracefully

## Code Structure

- `lambda_function.py`: Main Lambda handler
- `image_manager.py`: Core business logic for managing image operations
- `README.md`: This documentation file

## Dependencies

- boto3: AWS SDK for Python
- Python 3.8 or higher 