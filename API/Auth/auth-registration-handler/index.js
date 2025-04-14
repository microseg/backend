const { AuthService } = require('./src/services/AuthService');
const { ResponseBuilder } = require('./src/utils/ResponseBuilder');

class AuthHandler {
    constructor(stage) {
        this.authService = new AuthService(stage);
    }

    // Get path from event
    getPath(event) {
        console.log('Getting path from event:', JSON.stringify(event));
        
        // For HTTP API (API Gateway V2)
        if (event.requestContext?.routeKey) {
            const path = event.requestContext.routeKey.split(' ')[1];
            if (path === '/register' || path === '/verify-email' || path === '/resend-verification') {
                console.log('Found path in routeKey:', path.substring(1));
                return path.substring(1);
            }
        }

        // Extract from resource path
        if (event.resource) {
            const pathParts = event.resource.split('/');
            const lastPart = pathParts[pathParts.length - 1];
            if (lastPart === 'register' || lastPart === 'verify-email' || lastPart === 'resend-verification') {
                console.log('Found path in resource:', lastPart);
                return lastPart;
            }
        }

        // Extract from raw path
        if (event.path) {
            const pathParts = event.path.split('/');
            const lastPart = pathParts[pathParts.length - 1];
            if (lastPart === 'register' || lastPart === 'verify-email' || lastPart === 'resend-verification') {
                console.log('Found path in path:', lastPart);
                return lastPart;
            }
        }

        // If no valid path found but has specific fields, determine the operation
        if (event.email) {
            if (event.password && event.username) {
                console.log('Direct invocation detected for registration');
                return 'register';
            } else if (event.verificationCode) {
                console.log('Direct invocation detected for verification');
                return 'verify-email';
            } else {
                console.log('Direct invocation detected for resend verification');
                return 'resend-verification';
            }
        }

        return null;
    }

    // Normalize path by removing stage and extra slashes
    normalizePath(path, event) {
        if (!path) return '';

        console.log('Normalizing path:', path);
        
        // Remove stage prefix if present
        const stage = this.getStage(event);
        if (stage) {
            const stagePrefix = `/${stage}/`;
            if (path.startsWith(stagePrefix)) {
                path = path.substring(stagePrefix.length);
                console.log('Path after removing stage:', path);
            }
        }

        // Remove leading and trailing slashes
        path = path.replace(/^\/+|\/+$/g, '');
        console.log('Path after removing slashes:', path);

        // Extract the last segment of the path
        const segments = path.split('/');
        const lastSegment = segments[segments.length - 1];
        console.log('Final path segment:', lastSegment);

        return lastSegment;
    }

    // Get stage from event
    getStage(event) {
        console.log('Getting stage from event:', JSON.stringify(event));
        
        // First check environment variable
        if (process.env.STAGE) {
            console.log('Found stage in environment variable:', process.env.STAGE);
            return process.env.STAGE;
        }
        
        // For HTTP API (API Gateway V2)
        if (event.requestContext?.stage) {
            console.log('Found stage in requestContext.stage:', event.requestContext.stage);
            return event.requestContext.stage;
        }

        // For REST API (API Gateway V1)
        if (event.requestContext?.path) {
            const pathParts = event.requestContext.path.split('/');
            if (pathParts.length > 1) {
                console.log('Found stage in path parts:', pathParts[1]);
                return pathParts[1];
            }
        }

        // From resource path
        if (event.resource) {
            const pathParts = event.resource.split('/');
            if (pathParts.length > 1) {
                console.log('Found stage in resource:', pathParts[1]);
                return pathParts[1];
            }
        }

        // Extract from path
        if (event.path) {
            const pathParts = event.path.split('/');
            if (pathParts.length > 1) {
                console.log('Found stage in path:', pathParts[1]);
                return pathParts[1];
            }
        }

        console.log('No stage found, defaulting to prod');
        return 'prod';
    }

    // Parse and validate request body
    parseBody(event) {
        try {
            console.log('Parsing body from event:', event);
            
            // If body is already an object, return it
            if (typeof event.body === 'object' && event.body !== null) {
                console.log('Body is already an object:', event.body);
                return event.body;
            }

            // If body is a string, try to parse it
            if (typeof event.body === 'string') {
                const body = JSON.parse(event.body);
                console.log('Parsed body from string:', body);
                return body;
            }

            // If this is a direct invocation with email
            if (event.email) {
                const body = { email: event.email };
                if (event.username) body.username = event.username;
                if (event.password) body.password = event.password;
                if (event.verificationCode) body.verificationCode = event.verificationCode;
                console.log('Using direct event fields:', body);
                return body;
            }

            // If no valid body found, return empty object
            console.log('No valid body found, returning empty object');
            return {};
        } catch (error) {
            console.error('Error parsing body:', error);
            throw new Error('Invalid request body');
        }
    }

    // Handle registration request
    async handleRegistration(body) {
        console.log('Processing registration request with body:', body);
        return await this.authService.register(
            body.email,
            body.username,
            body.password
        );
    }

    // Handle email verification request
    async handleEmailVerification(body) {
        console.log('Processing email verification request with body:', body);
        return await this.authService.verifyEmail(
            body.email,
            body.verificationCode
        );
    }

    // Handle verification code resend request
    async handleResendVerification(body) {
        console.log('Processing verification code resend request with body:', body);
        
        // Additional validation for email
        if (!body.email) {
            throw new Error('Email is required for resending verification code');
        }
        
        // Ensure email is a string and not empty
        if (typeof body.email !== 'string' || body.email.trim() === '') {
            throw new Error('Invalid email format');
        }

        return await this.authService.resendVerification(
            body.email.trim()
        );
    }

    // Main request handler
    async handleRequest(event) {
        try {
            console.log('Received event:', JSON.stringify(event));
            
            const path = this.getPath(event);
            if (!path) {
                return ResponseBuilder.error({
                    message: 'Invalid request path',
                    statusCode: 400
                });
            }

            const body = this.parseBody(event);
            console.log('Processing request for path:', path, 'with body:', body);

            let result;
            
            switch (path) {
                case 'register':
                    result = await this.handleRegistration(body);
                    break;
                    
                case 'verify-email':
                    result = await this.handleEmailVerification(body);
                    break;
                    
                case 'resend-verification':
                    if (!body.email) {
                        return ResponseBuilder.error({
                            message: 'Email is required',
                            statusCode: 400
                        });
                    }
                    result = await this.handleResendVerification(body);
                    break;
                    
                default:
                    console.log('Invalid path requested:', path);
                    return ResponseBuilder.error({
                        message: `Invalid endpoint: ${path}`,
                        statusCode: 404
                    });
            }

            return ResponseBuilder.success(result);
        } catch (error) {
            console.error('Error in handler:', error);
            return ResponseBuilder.error({
                message: error.message,
                statusCode: error.statusCode || 500
            });
        }
    }
}

// Lambda handler function
exports.handler = async (event) => {
    try {
        const handler = new AuthHandler(null);
        const stage = handler.getStage(event);
        console.log('Using stage:', stage);
        return await new AuthHandler(stage).handleRequest(event);
    } catch (error) {
        console.error('Fatal error:', error);
        return ResponseBuilder.error({
            message: 'Internal server error',
            statusCode: 500
        });
    }
}; 