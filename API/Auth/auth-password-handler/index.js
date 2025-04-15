const PasswordService = require('./src/services/PasswordService');
const ResponseBuilder = require('./src/utils/ResponseBuilder');

exports.handler = async (event) => {
  try {
    console.log('Starting Password Handler Lambda execution');
    console.log('Received event:', JSON.stringify(event, null, 2));

    // Handle input: Support both direct Lambda testing and API Gateway requests
    let requestBody = event;
    let path = '';
    
    // If the request comes from API Gateway, event will contain a body field and path
    if (event.body) {
      try {
        console.log('Parsing API Gateway body');
        requestBody = JSON.parse(event.body);
        path = event.path || '';
        console.log('Parsed body:', JSON.stringify(requestBody, null, 2));
        console.log('Request path:', path);
      } catch (e) {
        console.error('JSON parsing error:', e);
        return ResponseBuilder.error('Invalid JSON format in request body', 400);
      }
    }

    // Create password service instance
    console.log('Initializing password service');
    const passwordService = new PasswordService();
    
    // Route to appropriate handler based on path
    if (path.includes('/forgot-password') || event.forgotPassword) {
      console.log('Processing forgot password request');
      
      // Validate required parameters
      const { email } = requestBody;
      if (!email) {
        console.log('Missing email parameter');
        return ResponseBuilder.error('Email is required', 400);
      }
      
      const result = await passwordService.forgotPassword(email);
      return ResponseBuilder.success(result);
    } 
    else if (path.includes('/reset-password') || event.resetPassword) {
      console.log('Processing reset password request');
      
      // Validate required parameters
      const { email, verificationCode, newPassword } = requestBody;
      if (!email || !verificationCode || !newPassword) {
        console.log('Missing required parameters');
        return ResponseBuilder.error('Email, verification code, and new password are required', 400);
      }
      
      const result = await passwordService.resetPassword(email, verificationCode, newPassword);
      return ResponseBuilder.success(result);
    }
    else {
      console.log('Unknown request type');
      return ResponseBuilder.error('Invalid request type. Use /forgot-password or /reset-password endpoints', 400);
    }
  } catch (error) {
    // Log error details
    console.error('Password handler error:', error);
    console.error('Error stack:', error.stack);
    
    // Return appropriate error response
    return ResponseBuilder.error(
      error.message || 'An error occurred during password operation',
      error.statusCode || 400
    );
  }
}; 