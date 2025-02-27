# CreateUserFolder API

这个 Lambda 函数用于在用户成功注册后，在 S3 存储桶中创建用户专属的文件夹。

## 功能

- 接收用户 ID（Cognito sub）
- 在指定的 S3 存储桶中创建用户专属文件夹
- 返回创建状态和文件夹路径

## 输入格式

```json
{
    "user_id": "user_sub_from_cognito"
}
```

## 输出格式

成功响应：
```json
{
    "statusCode": 200,
    "headers": {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*"
    },
    "body": {
        "message": "User folder created successfully",
        "folder_key": "users/user_sub_from_cognito/"
    }
}
```

错误响应：
```json
{
    "statusCode": 400/500,
    "body": {
        "error": "错误信息"
    }
}
```

## 环境变量

- `BUCKET_NAME`: S3 存储桶名称

## 所需权限

- `s3:PutObject`: 用于在 S3 存储桶中创建文件夹 