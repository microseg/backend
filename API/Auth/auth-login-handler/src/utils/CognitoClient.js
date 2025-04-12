const { 
  CognitoIdentityProviderClient, 
  InitiateAuthCommand,
  GetUserCommand
} = require('@aws-sdk/client-cognito-identity-provider');
const crypto = require('crypto');

class CognitoClient {
  constructor() {
    this.client = new CognitoIdentityProviderClient({
      region: process.env.REGION
    });
    this.clientId = process.env.APP_CLIENT_ID;
    this.clientSecret = process.env.APP_CLIENT_SECRET;
  }

  calculateSecretHash(username) {
    const message = username + this.clientId;
    const key = Buffer.from(this.clientSecret, 'utf8');
    return crypto
      .createHmac('SHA256', key)
      .update(message)
      .digest('base64');
  }

  async initiateAuth(username, password) {
    const secretHash = this.calculateSecretHash(username);
    
    const command = new InitiateAuthCommand({
      AuthFlow: 'USER_PASSWORD_AUTH',
      ClientId: this.clientId,
      AuthParameters: {
        USERNAME: username,
        PASSWORD: password,
        SECRET_HASH: secretHash
      }
    });

    return await this.client.send(command);
  }

  async getUserInfo(accessToken) {
    const command = new GetUserCommand({
      AccessToken: accessToken
    });

    return await this.client.send(command);
  }
}

module.exports = CognitoClient; 