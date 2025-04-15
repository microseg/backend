const CognitoClient = require('../utils/CognitoClient');

class PasswordService {
  constructor() {
    console.log('Initializing PasswordService');
    this.cognitoClient = new CognitoClient();
  }

  /**
   * Initiates the forgot password flow by sending a verification code to the user's email
   * @param {string} email - The user's email address
   * @returns {Object} - Response indicating success
   */
  async forgotPassword(email) {
    try {
      console.log('PasswordService: Starting forgot password process for:', email);
      
      // Perform forgot password initiation
      await this.cognitoClient.forgotPassword(email);
      console.log('PasswordService: Verification code sent successfully');
      
      // Return success response
      return {
        message: 'Password reset code has been sent to your email',
        email: email
      };
    } catch (error) {
      console.error('PasswordService: Error during forgot password:', error);
      console.error('PasswordService: Error stack:', error.stack);

      // Handle specific error types
      if (error.name === 'UserNotFoundException') {
        throw new Error('We could not find an account with that email address');
      } else if (error.name === 'LimitExceededException') {
        throw new Error('Too many attempts. Please try again later');
      } else if (error.name === 'InvalidParameterException') {
        throw new Error('Invalid email format');
      }
      
      // Handle other errors
      throw new Error(error.message || 'Failed to send verification code. Please try again later');
    }
  }

  /**
   * Resets the user's password using the verification code
   * @param {string} email - The user's email address
   * @param {string} verificationCode - The verification code sent to the user's email
   * @param {string} newPassword - The new password
   * @returns {Object} - Response indicating success
   */
  async resetPassword(email, verificationCode, newPassword) {
    try {
      console.log('PasswordService: Starting password reset for:', email);
      
      // Validate password strength
      this.validatePassword(newPassword);
      
      // Confirm password reset
      await this.cognitoClient.confirmForgotPassword(email, verificationCode, newPassword);
      console.log('PasswordService: Password reset successful');
      
      // Return success response
      return {
        message: 'Password has been reset successfully. You can now login with your new password'
      };
    } catch (error) {
      console.error('PasswordService: Error during password reset:', error);
      console.error('PasswordService: Error stack:', error.stack);

      // Handle specific error types
      if (error.name === 'CodeMismatchException') {
        throw new Error('Invalid verification code');
      } else if (error.name === 'ExpiredCodeException') {
        throw new Error('Verification code has expired. Please request a new one');
      } else if (error.name === 'UserNotFoundException') {
        throw new Error('User not found');
      } else if (error.name === 'InvalidPasswordException') {
        throw new Error('Password does not meet requirements. It must include uppercase, lowercase, numbers, and special characters');
      } else if (error.name === 'LimitExceededException') {
        throw new Error('Too many attempts. Please try again later');
      } else if (error.name === 'PasswordValidationError') {
        throw error; // Rethrow our custom validation error
      }
      
      // Handle other errors
      throw new Error(error.message || 'Failed to reset password. Please try again later');
    }
  }

  /**
   * Validates password strength
   * @param {string} password - The password to validate
   */
  validatePassword(password) {
    // Check length
    if (password.length < 8) {
      const error = new Error('Password must be at least 8 characters long');
      error.name = 'PasswordValidationError';
      throw error;
    }

    // Check for uppercase, lowercase, number and special character
    const hasUppercase = /[A-Z]/.test(password);
    const hasLowercase = /[a-z]/.test(password);
    const hasNumber = /[0-9]/.test(password);
    const hasSpecial = /[^A-Za-z0-9]/.test(password);

    if (!hasUppercase || !hasLowercase || !hasNumber || !hasSpecial) {
      const error = new Error('Password must include uppercase, lowercase, numbers, and special characters');
      error.name = 'PasswordValidationError';
      throw error;
    }
  }
}

module.exports = PasswordService; 