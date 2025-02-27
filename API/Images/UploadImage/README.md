# Upload Image API

This Lambda function handles image uploads to S3 with proper folder structure.

## Input Format

```json
{
    "prefix": "folder/",  // Optional folder prefix
    "image_key": "filename.jpg",  // Required: name of the file
    "image_data": "base64_encoded_string"  // Required: base64 encoded image data
}
```

## Output Format

Success (200):
```json
{
    "message": "Image uploaded successfully",
    "key": "users/folder/filename.jpg"
}
```

Error (400):
```json
{
    "message": "Missing required parameters: image_key or image_data"
}
```

or

```json
{
    "message": "Invalid base64 image data"
}
```

Error (500):
```json
{
    "message": "Failed to upload image: [error details]"
}
```

## Environment Variables

- `BUCKET_NAME`: The S3 bucket name where images will be stored

## Folder Structure

Images are stored in the following structure:
`users/[prefix]/[image_key]`

Where:
- `users/` is the base prefix for all user uploads
- `[prefix]` is the optional folder prefix from the request
- `[image_key]` is the filename of the uploaded image 