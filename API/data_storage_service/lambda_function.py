import logging
import json
from handlers import RequestHandler

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a single instance of the request handler
request_handler = RequestHandler()

def lambda_handler(event, context):
    """
    AWS Lambda handler function that delegates to appropriate handlers
    based on the method specified in the event.
    
    Accepts JSON input with the following structure:
    {
        "method": "POST|GET|UPDATE|DELETE|SYNC",
        "model": "users|products|etc",
        "data": {...} or [{...}, {...}],  // Single object or array for batch operations
        "id": 123 or "some-id",           // Primary ID or another identifier (optional)
        "filter": {...},                  // For GET with filters (optional)
        "id_field": "id",                 // Custom ID field to use (optional, defaults to "id")
        "db_type": "postgres|mysql|dynamodb"   // Database type (optional, defaults to postgres)
    }
    
    Args:
        event (dict): AWS Lambda event with operation details
        context (LambdaContext): AWS Lambda context object
        
    Returns:
        dict: Response with status code and operation results
    """
    try:
        # Log the incoming event (truncated to avoid huge logs)
        logger.info(f"Received event: {str(event)[:1000]}" + ("..." if len(str(event)) > 1000 else ""))
        
        # Handle request using the RequestHandler
        return request_handler.handle_request(event)
        
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unhandled exception in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'success': False,
                'error': f"Internal server error: {str(e)}"
            }
        }