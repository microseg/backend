# Delete Image API

This Lambda function is used to delete specified images from the S3 bucket.

## Input Format

```json
{
    "image_key": "users/folder/filename.jpg"  // Required: full path of the image to delete
}
```

## Output Format

Success (200):
```json
{
    "message": "Image deleted successfully",
    "key": "users/folder/filename.jpg",
    "bucket": "bucket-name"
}
```

Error (400):
```json
{
    "message": "Missing required parameter: image_key"
}
```

Error (404):
```json
{
    "message": "Image to delete does not exist: users/folder/filename.jpg"
}
```

Error (500):
```json
{
    "message": "Failed to delete image: [error details]"
}
```

## Environment Variables

- `BUCKET_NAME`: Name of the S3 bucket where images are stored

## Security

- Function requires IAM permissions to delete objects from S3
- It is recommended to verify user permissions before calling this API to ensure users can only delete their own images

## Error Handling

1. Validates input parameters
2. Verifies object existence before deletion
3. Validates deletion success
4. Handles all possible S3 errors
5. Provides detailed error messages and logging 