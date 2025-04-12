const CognitoClient = require('../utils/CognitoClient');
const User = require('../models/User');

class AuthService {
  constructor() {
    console.log('Initializing AuthService');
    this.cognitoClient = new CognitoClient();
  }

  async login(username, password) {
    try {
      console.log('AuthService: Starting login process for user:', username);
      
      // Perform authentication
      console.log('AuthService: Initiating Cognito authentication');
      const authResponse = await this.cognitoClient.initiateAuth(username, password);
      console.log('AuthService: Cognito authentication successful');
      
      // Get user information
      console.log('AuthService: Fetching user information');
      const userInfo = await this.cognitoClient.getUserInfo(authResponse.AuthenticationResult.AccessToken);
      console.log('AuthService: User information retrieved');
      
      const user = User.fromCognitoUser(userInfo);
      console.log('AuthService: User model created');

      // Return authentication result and user information
      console.log('AuthService: Preparing response');
      return {
        tokens: {
          accessToken: authResponse.AuthenticationResult.AccessToken,
          refreshToken: authResponse.AuthenticationResult.RefreshToken,
          idToken: authResponse.AuthenticationResult.IdToken
        },
        user: user.toJSON()
      };
    } catch (error) {
      console.error('AuthService: Error during login:', error);
      console.error('AuthService: Error stack:', error.stack);

      // Handle specific error types
      if (error.name === 'UserNotConfirmedException') {
        throw new Error('Your account needs email verification. Please verify your email first');
      } else if (error.name === 'NotAuthorizedException') {
        throw new Error('Incorrect email or password');
      } else if (error.name === 'UserNotFoundException') {
        throw new Error('Account not found. Please check your email');
      }
      
      // Handle other errors
      throw new Error(error.message || 'Login failed. Please try again later');
    }
  }
}

module.exports = AuthService; 