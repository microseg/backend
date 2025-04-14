const { 
    CognitoIdentityProviderClient,
    SignUpCommand,
    ConfirmSignUpCommand,
    ResendConfirmationCodeCommand,
    AdminGetUserCommand
} = require('@aws-sdk/client-cognito-identity-provider');
const crypto = require('crypto');

class CognitoClient {
    constructor(stage = 'dev') {
        if (!stage) {
            console.warn('No stage provided, defaulting to dev');
            stage = 'dev';
        }
        
        this.stage = stage.toLowerCase();
        console.log('Initializing CognitoClient for stage:', this.stage);
        
        this.validateEnvironmentVariables();

        const region = process.env[`${this.stage.toUpperCase()}_REGION`];
        this.client = new CognitoIdentityProviderClient({ region });
        this.clientId = process.env[`${this.stage.toUpperCase()}_APP_CLIENT_ID`];
        this.clientSecret = process.env[`${this.stage.toUpperCase()}_APP_CLIENT_SECRET`];
        this.userPoolId = process.env[`${this.stage.toUpperCase()}_USER_POOL_ID`];
        
        console.log('CognitoClient initialized with region:', region);
    }

    validateEnvironmentVariables() {
        const stageUpper = this.stage.toUpperCase();
        const requiredEnvVars = [
            `${stageUpper}_REGION`,
            `${stageUpper}_APP_CLIENT_ID`,
            `${stageUpper}_APP_CLIENT_SECRET`,
            `${stageUpper}_USER_POOL_ID`
        ];

        const missingEnvVars = requiredEnvVars.filter(
            envVar => !process.env[envVar]
        );

        if (missingEnvVars.length > 0) {
            const error = new Error(
                `Missing required environment variables for ${this.stage} environment: ${missingEnvVars.join(', ')}`
            );
            console.error('Environment validation failed:', error);
            throw error;
        }
        
        console.log(`Environment variables validated for stage: ${this.stage}`);
    }

    calculateSecretHash(username) {
        if (!username) {
            throw new Error('Username is required for secret hash calculation');
        }

        try {
            const message = username + this.clientId;
            const hmac = crypto.createHmac('SHA256', this.clientSecret);
            hmac.update(message);
            return hmac.digest('base64');
        } catch (error) {
            console.error('Error calculating secret hash:', error);
            throw new Error('Failed to calculate secret hash');
        }
    }

    async signUp(email, username, password) {
        if (!email || !username || !password) {
            throw new Error('Email, username, and password are required for sign up');
        }

        console.log(`Using ${this.stage} environment for registration`);
        const secretHash = this.calculateSecretHash(email);
        
        const command = new SignUpCommand({
            ClientId: this.clientId,
            Username: email,
            Password: password,
            SecretHash: secretHash,
            UserAttributes: [
                {
                    Name: 'email',
                    Value: email
                },
                {
                    Name: 'preferred_username',
                    Value: username
                }
            ]
        });

        return await this.client.send(command);
    }

    async confirmSignUp(email, code) {
        if (!email || !code) {
            throw new Error('Email and verification code are required');
        }

        console.log(`Using ${this.stage} environment for email verification`);
        const secretHash = this.calculateSecretHash(email);
        
        const command = new ConfirmSignUpCommand({
            ClientId: this.clientId,
            Username: email,
            ConfirmationCode: code,
            SecretHash: secretHash
        });

        return await this.client.send(command);
    }

    async resendConfirmationCode(email) {
        if (!email) {
            throw new Error('Email is required');
        }

        console.log(`Using ${this.stage} environment for resending verification code`);
        const secretHash = this.calculateSecretHash(email);
        
        const command = new ResendConfirmationCodeCommand({
            ClientId: this.clientId,
            Username: email,
            SecretHash: secretHash
        });

        return await this.client.send(command);
    }

    async getUserStatus(email) {
        if (!email) {
            throw new Error('Email is required');
        }

        try {
            console.log(`Getting user status for email: ${email} in ${this.stage} environment`);
            
            const command = new AdminGetUserCommand({
                UserPoolId: this.userPoolId,
                Username: email
            });

            try {
                const response = await this.client.send(command);
                console.log('User status response:', response);
                return response.UserStatus;
            } catch (adminError) {
                console.log('AdminGetUser error:', adminError);
                if (adminError.name === 'UserNotFoundException') {
                    return null;
                }
                throw adminError;
            }
        } catch (error) {
            console.error('Error getting user status:', error);
            throw error;
        }
    }
}

module.exports = { CognitoClient }; 