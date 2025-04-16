const { LambdaClient: AwsLambdaClient, InvokeCommand } = require('@aws-sdk/client-lambda');

class UserLambdaClient {
    constructor(stage = 'dev') {
        this.lambdaClient = new AwsLambdaClient({ region: process.env.AWS_REGION || 'us-east-1' });
        this.stage = stage;
    }

    /**
     * Invoke a Lambda function
     * @param {string} functionName - The name of the Lambda function to invoke
     * @param {object} payload - The data to send to the Lambda function
     * @returns {Promise<object>} - The response from the Lambda function
     */
    async invokeLambda(functionName, payload) {
        try {
            console.log(`Invoking lambda ${functionName} with payload:`, payload);
            
            // First check if TARGET_LAMBDA_ARN environment variable exists
            const targetLambdaArn = process.env.TARGET_LAMBDA_ARN;
            
            // If it exists, use the complete ARN
            if (targetLambdaArn) {
                console.log(`Using target Lambda ARN from environment: ${targetLambdaArn}`);
                
                const params = {
                    FunctionName: targetLambdaArn,
                    InvocationType: 'Event', // Asynchronous invocation
                    Payload: JSON.stringify(payload)
                };
    
                const command = new InvokeCommand(params);
                const response = await this.lambdaClient.send(command);
                
                console.log(`Lambda invocation response:`, response);
                return response;
            }
            
            // Otherwise use the original logic
            // Allow overriding function name via environment variables
            const envFunctionName = process.env[`LAMBDA_${functionName.replace(/-/g, '_').toUpperCase()}`];
            
            // Function name may need stage prefix, unless environment variable already specifies the complete name
            const qualifiedFunctionName = envFunctionName || `${functionName}-${this.stage}`;
            
            console.log(`Using qualified function name: ${qualifiedFunctionName}`);
            
            const params = {
                FunctionName: qualifiedFunctionName,
                InvocationType: 'Event', // Asynchronous invocation
                Payload: JSON.stringify(payload)
            };

            const command = new InvokeCommand(params);
            const response = await this.lambdaClient.send(command);
            
            console.log(`Lambda invocation response:`, response);
            return response;
        } catch (error) {
            console.error(`Error invoking lambda ${functionName}:`, error);
            throw error;
        }
    }
}

module.exports = { UserLambdaClient }; 