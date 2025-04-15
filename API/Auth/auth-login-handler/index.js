const AuthService = require('./src/services/AuthService');
const ResponseBuilder = require('./src/utils/ResponseBuilder');

exports.handler = async (event) => {
  try {
    console.log('Starting Lambda execution');
    console.log('Received event:', JSON.stringify(event, null, 2));

    // Handle input: Support both direct Lambda testing and API Gateway requests
    let requestBody = event;
    console.log('Initial request body:', JSON.stringify(requestBody, null, 2));
    
    // If the request comes from API Gateway, event will contain a body field
    if (event.body) {
      try {
        console.log('Parsing API Gateway body');
        requestBody = JSON.parse(event.body);
        console.log('Parsed body:', JSON.stringify(requestBody, null, 2));
      } catch (e) {
        console.error('JSON parsing error:', e);
        return ResponseBuilder.error('Invalid JSON format in request body', 400);
      }
    }

    // Validate required parameters
    console.log('Validating parameters');
    const { username, password } = requestBody;
    if (!username || !password) {
      console.log('Missing required parameters');
      return ResponseBuilder.error('Username and password are required', 400);
    }

    // Create auth service instance and perform login
    console.log('Initializing auth service');
    const authService = new AuthService();
    console.log('Starting login process');
    const result = await authService.login(username, password);
    console.log('Login successful');

    // Return success response
    console.log('Preparing success response');
    return ResponseBuilder.success(result);
  } catch (error) {
    // Log error details
    console.error('Login error:', error);
    console.error('Error stack:', error.stack);
    
    // Special handling for email not verified case
    if (error.code === 'EMAIL_NOT_VERIFIED') {
      console.log('User email not verified:', error.username);
      return ResponseBuilder.error(
        error.message,
        error.statusCode,
        {
          code: error.code,
          username: error.username
        }
      );
    }
    
    // Return appropriate error response
    return ResponseBuilder.error(
      error.message || 'An error occurred during login',
      error.statusCode || 400
    );
  }
}; 