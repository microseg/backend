# ReadImage Lambda 函数

这个Lambda函数用于从S3存储桶中获取图片。

## 配置

### Lambda 函数别名

函数使用 Lambda 别名来确定当前环境：
- `$LATEST` 或无别名: 默认使用 'dev' 环境
- `dev`: 开发环境
- `staging`: 测试环境
- `prod`: 生产环境

### Parameter Store 参数

在 AWS Systems Manager Parameter Store 中需要设置以下参数：

- `/user-image-bucket-{stage}`: S3存储桶名称
  - 例如：
    - `/user-image-bucket-dev`: 'test-matsight-dev'
    - `/user-image-bucket-staging`: 'test-matsight-staging'
    - `/user-image-bucket-prod`: 'test-matsight-prod'

### 环境变量（备用）

当Parameter Store不可用时使用：
- `BUCKET_NAME`: S3存储桶名称

## API 参数

通过API Gateway调用时，需要提供以下查询参数：

- `image_key`: 图片在S3中的键值（必需）

## 返回格式

成功响应：
```json
{
    "statusCode": 200,
    "headers": {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*"
    },
    "body": {
        "url": "预签名URL",
        "expires_in": 3600
    }
}
```

错误响应：
```json
{
    "statusCode": 400/404/500,
    "body": {
        "error": "错误信息"
    }
}
```

## Lambda函数权限

需要确保Lambda函数有以下IAM权限：

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject"
            ],
            "Resource": [
                "arn:aws:s3:::test-matsight-dev/*",
                "arn:aws:s3:::test-matsight-staging/*",
                "arn:aws:s3:::test-matsight-prod/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "ssm:GetParameter",
                "ssm:GetParameters"
            ],
            "Resource": "arn:aws:ssm:*:*:parameter/user-image-bucket*"
        }
    ]
}
```

## 部署说明

1. 创建Lambda函数版本
```bash
aws lambda publish-version --function-name ReadImage
```

2. 创建或更新Lambda别名
```bash
# 开发环境
aws lambda create-alias --function-name ReadImage --name dev --function-version 1

# 测试环境
aws lambda create-alias --function-name ReadImage --name staging --function-version 1

# 生产环境
aws lambda create-alias --function-name ReadImage --name prod --function-version 1
```

## 使用示例

API Gateway URL调用示例：
```
GET https://your-api-gateway-url/stage/getImage?image_key=path/to/your/image.jpg
``` 