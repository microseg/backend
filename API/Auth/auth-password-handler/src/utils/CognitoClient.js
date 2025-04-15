const { 
  CognitoIdentityProviderClient, 
  ForgotPasswordCommand,
  ConfirmForgotPasswordCommand
} = require('@aws-sdk/client-cognito-identity-provider');
const crypto = require('crypto');

class CognitoClient {
  constructor() {
    // Determine environment based on STAGE variable
    const stage = process.env.STAGE || 'dev';
    const isProduction = stage.toLowerCase() === 'prod' || stage.toLowerCase() === 'production';
    const prefix = isProduction ? 'PROD_' : 'DEV_';
    
    console.log(`CognitoClient: Operating in ${isProduction ? 'PRODUCTION' : 'DEVELOPMENT'} environment`);
    
    // Set up client with environment-specific configuration
    this.client = new CognitoIdentityProviderClient({
      region: process.env[`${prefix}REGION`] || process.env.REGION
    });
    
    // Store configuration
    this.clientId = process.env[`${prefix}APP_CLIENT_ID`] || process.env.APP_CLIENT_ID;
    this.clientSecret = process.env[`${prefix}APP_CLIENT_SECRET`] || process.env.APP_CLIENT_SECRET;
    this.userPoolId = process.env[`${prefix}USER_POOL_ID`];
    
    // Log configuration (but not secrets)
    console.log(`CognitoClient: Using Region: ${process.env[`${prefix}REGION`] || process.env.REGION}`);
    console.log(`CognitoClient: Using Client ID: ${this.clientId}`);
    console.log(`CognitoClient: Using User Pool ID: ${this.userPoolId}`);
  }

  calculateSecretHash(username) {
    const message = username + this.clientId;
    const key = Buffer.from(this.clientSecret, 'utf8');
    return crypto
      .createHmac('SHA256', key)
      .update(message)
      .digest('base64');
  }

  async forgotPassword(username) {
    console.log('CognitoClient: initiating forgot password for:', username);
    
    const params = {
      ClientId: this.clientId,
      Username: username
    };
    
    // Add secret hash if client secret is configured
    if (this.clientSecret) {
      params.SecretHash = this.calculateSecretHash(username);
    }
    
    const command = new ForgotPasswordCommand(params);
    
    try {
      const response = await this.client.send(command);
      console.log('CognitoClient: forgot password initiated successfully');
      return response;
    } catch (error) {
      console.error('CognitoClient: forgot password error:', error);
      throw error;
    }
  }

  async confirmForgotPassword(username, confirmationCode, newPassword) {
    console.log('CognitoClient: confirming forgot password for:', username);
    
    const params = {
      ClientId: this.clientId,
      Username: username,
      ConfirmationCode: confirmationCode,
      Password: newPassword
    };
    
    // Add secret hash if client secret is configured
    if (this.clientSecret) {
      params.SecretHash = this.calculateSecretHash(username);
    }
    
    const command = new ConfirmForgotPasswordCommand(params);
    
    try {
      const response = await this.client.send(command);
      console.log('CognitoClient: password reset confirmed successfully');
      return response;
    } catch (error) {
      console.error('CognitoClient: confirm forgot password error:', error);
      throw error;
    }
  }
}

module.exports = CognitoClient; 