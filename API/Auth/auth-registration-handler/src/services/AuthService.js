const { CognitoClient } = require('../utils/CognitoClient');
const { UserLambdaClient } = require('../utils/LambdaClient');

class AuthService {
    constructor(stage = 'dev') {
        this.cognitoClient = new CognitoClient(stage);
        this.lambdaClient = new UserLambdaClient(stage);
        this.stage = stage; // Store stage for later use
    }

    async register(email, username, password) {
        try {
            console.log('Starting registration process for email:', email);
            console.log('Current stage:', this.stage);
            
            // Validate input
            if (!email || !username || !password) {
                throw new Error('Missing required fields');
            }

            // Check if user exists and get status
            try {
                const userStatus = await this.cognitoClient.getUserStatus(email);
                console.log('User status check result:', userStatus);

                if (userStatus === 'UNCONFIRMED') {
                    // User exists but is not verified
                    return {
                        message: 'This email is already registered but not verified. A new verification code has been sent to your email.',
                        status: 'UNCONFIRMED'
                    };
                } else if (userStatus) {
                    // User exists and is confirmed
                    throw new Error('Email already registered');
                }
            } catch (statusError) {
                console.log('Status check error:', statusError);
                if (statusError.message === 'Email already registered') {
                    throw statusError;
                }
                // If there was an error checking status but it's not about existing user,
                // continue with registration attempt
            }

            // Attempt to register the user
            try {
                const signUpResult = await this.cognitoClient.signUp(email, username, password);
                console.log('User registered in Cognito:', signUpResult.UserSub);

                // Only call the user creation Lambda if the stage is 'prod'
                if (this.stage === 'prod') {
                    try {
                        console.log('Stage is prod, calling user creation Lambda');
                        // Set the target Lambda ARN (defaults to getting from environment variables)
                        process.env.TARGET_LAMBDA_ARN = 'arn:aws:lambda:us-east-1:043309364810:function:NewFolder:prod';
                        
                        // Get the target Lambda function name from environment variables, default to user-creation-handler
                        const targetFunction = process.env.USER_CREATION_LAMBDA || 'user-creation-handler';
                        
                        await this.lambdaClient.invokeLambda(targetFunction, {
                            user_id: signUpResult.UserSub
                        });
                        console.log(`Successfully called Lambda with user_id:`, signUpResult.UserSub);
                    } catch (lambdaError) {
                        // If Lambda call fails, log the error but don't affect the registration process
                        console.error('Error calling Lambda function:', lambdaError);
                    }
                } else {
                    console.log(`Stage is ${this.stage}, skipping user creation Lambda call`);
                }

                return {
                    message: 'Registration successful. Please check your email for verification code.',
                    userId: signUpResult.UserSub
                };
            } catch (signUpError) {
                console.log('Sign up error:', signUpError);
                
                if (signUpError.name === 'UsernameExistsException') {
                    // If we get here, the user exists but we couldn't get status earlier
                    // Try to resend verification code
                    await this.cognitoClient.resendConfirmationCode(email);
                    return {
                        message: 'This email is already registered but not verified. A new verification code has been sent to your email.',
                        status: 'UNCONFIRMED'
                    };
                }
                
                if (signUpError.name === 'InvalidPasswordException') {
                    throw new Error('Password does not meet requirements');
                }

                throw signUpError;
            }
        } catch (error) {
            console.error('Registration error:', error);
            throw error;
        }
    }

    async verifyEmail(email, code) {
        try {
            console.log('Starting email verification for:', email);
            
            if (!email || !code) {
                throw new Error('Missing required fields');
            }

            await this.cognitoClient.confirmSignUp(email, code);
            console.log('Email verification successful for:', email);

            return {
                message: 'Email verification successful'
            };
        } catch (error) {
            console.error('Email verification error:', error);
            
            if (error.name === 'CodeMismatchException') {
                throw new Error('Invalid verification code');
            }
            if (error.name === 'ExpiredCodeException') {
                throw new Error('Verification code has expired');
            }
            throw error;
        }
    }

    async resendVerification(email) {
        try {
            console.log('Resending verification code for:', email);
            
            if (!email) {
                throw new Error('Email is required');
            }

            await this.cognitoClient.resendConfirmationCode(email);
            console.log('Verification code resent successfully to:', email);

            return {
                message: 'A new verification code has been sent to your email. Please check your inbox and spam folder.',
                email: email
            };
        } catch (error) {
            console.error('Resend verification error:', error);
            
            if (error.name === 'UserNotFoundException') {
                throw new Error('User not found');
            }
            if (error.name === 'LimitExceededException') {
                throw new Error('Too many attempts, please try again later');
            }
            throw error;
        }
    }
}

module.exports = { AuthService }; 