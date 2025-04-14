class ResponseBuilder {
    static success(data) {
        return {
            statusCode: 200,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': true,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                success: true,
                data
            })
        };
    }

    static error({ message, statusCode = 400 }) {
        const response = {
            statusCode: statusCode,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': true,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                success: false,
                error: {
                    message,
                    code: statusCode
                }
            }),
            isBase64Encoded: false
        };
        
        console.log('Error response:', response);
        return response;
    }
}

module.exports = { ResponseBuilder }; 