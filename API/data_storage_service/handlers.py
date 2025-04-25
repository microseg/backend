import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from models import DatabaseOperationService

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class BaseHandler:
    """Base handler class that all method handlers inherit from"""
    
    def __init__(self, db_service: DatabaseOperationService):
        """
        Initialize the handler with database service
        
        Args:
            db_service (DatabaseOperationService): Database operations service
        """
        self.db_service = db_service
    
    def validate_common_params(self, model: str, method: str) -> Optional[Dict[str, Any]]:
        """
        Validate common parameters for all methods
        
        Args:
            model (str): The model name
            method (str): The HTTP method
            
        Returns:
            Optional[Dict[str, Any]]: Error response or None if validation passes
        """
        if not model:
            return error_response(400, "Missing required parameter: 'model'")
        return None
    
    def handle(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle the request - must be implemented by child classes
        
        Args:
            event (Dict[str, Any]): The event data
            
        Returns:
            Dict[str, Any]: Response object
        """
        raise NotImplementedError("Subclasses must implement handle method")


class PostHandler(BaseHandler):
    """Handler for POST (create) operations"""
    
    def handle(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle POST requests
        
        Args:
            event (Dict[str, Any]): The event data
            
        Returns:
            Dict[str, Any]: Response object
        """
        model = event.get('model', '')
        data = event.get('data', {})
        
        # Validate parameters
        validation_error = self.validate_common_params(model, 'POST')
        if validation_error:
            return validation_error
        
        # Auto-detect batch mode
        is_batch = isinstance(data, list)
        batch_data = data if is_batch else []
        
        if is_batch:
            # Batch create operation
            if not batch_data:
                return error_response(400, "Batch operation requires data as array")
            
            result = self.db_service.bulk_create_records(model, batch_data)
            return success_response(201, f"Created {len(result)} {model} records", result)
        else:
            # Single create operation
            if not data:
                return error_response(400, "POST operation requires 'data'")
            
            result = self.db_service.create_record(model, data)
            return success_response(201, f"Created {model} record", result)


class GetHandler(BaseHandler):
    """Handler for GET (read) operations"""
    
    def handle(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle GET requests
        
        Args:
            event (Dict[str, Any]): The event data
            
        Returns:
            Dict[str, Any]: Response object
        """
        model = event.get('model', '')
        record_id = event.get('id')
        filter_data = event.get('filter')
        id_field = event.get('id_field', 'id')
        
        # Validate parameters
        validation_error = self.validate_common_params(model, 'GET')
        if validation_error:
            return validation_error
        
        if record_id is not None:
            # If ID field is not the default, use filter instead
            if id_field != 'id':
                filter_data = {id_field: record_id}
                records = self.db_service.get_records(model, filter_data)
                if records:
                    return success_response(200, f"Retrieved {model} record", records[0])
                return error_response(404, f"{model} record with {id_field}={record_id} not found")
            else:
                # Use direct ID lookup
                result = self.db_service.get_record_by_id(model, record_id)
                if result:
                    return success_response(200, f"Retrieved {model} record", result)
                return error_response(404, f"{model} record with ID {record_id} not found")
        else:
            result = self.db_service.get_records(model, filter_data)
            return success_response(200, f"Retrieved {len(result)} {model} records", result)


class UpdateHandler(BaseHandler):
    """Handler for UPDATE operations"""
    
    def handle(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle UPDATE requests
        
        Args:
            event (Dict[str, Any]): The event data
            
        Returns:
            Dict[str, Any]: Response object
        """
        model = event.get('model', '')
        data = event.get('data', {})
        record_id = event.get('id')
        id_field = event.get('id_field', 'id')
        
        # Validate parameters
        validation_error = self.validate_common_params(model, 'UPDATE')
        if validation_error:
            return validation_error
        
        # Auto-detect batch mode
        is_batch = isinstance(data, list)
        batch_data = data if is_batch else []
        
        # For single updates with no record_id, extract ID from data and convert to batch format
        if not is_batch and record_id is None and isinstance(data, dict):
            if id_field in data or 'id' in data:
                logger.info(f"Extracting record identifier from data for UPDATE operation")
                is_batch = True
                batch_data = [data]
                data = {}
                logger.info(f"Converted single update to batch format with 1 item")
        
        if is_batch:
            # Batch update operation
            if not all(id_field in item or 'id' in item for item in batch_data):
                return error_response(400, f"Batch update requires '{id_field}' field in each item")
            
            updates = []
            
            for item in batch_data:
                # Extract ID using either specified id_field or default 'id'
                if id_field in item:
                    item_id_value = item.pop(id_field)
                    
                    # Handle custom ID field - need to find by that field first
                    if id_field != 'id':
                        logger.info(f"Looking for {model} records with {id_field}={item_id_value}")
                        records = self.db_service.get_records(model, {id_field: item_id_value})
                        logger.info(f"Found {len(records)} matching records")
                        
                        if records and len(records) > 0:
                            actual_id = records[0]['id']
                            logger.info(f"Using actual ID {actual_id} for record with {id_field}={item_id_value}")
                            updates.append((actual_id, item))
                        else:
                            logger.warning(f"No record found with {id_field}={item_id_value}, skipping update")
                    else:
                        # Direct ID update
                        updates.append((item_id_value, item))
                elif 'id' in item:
                    # Default ID field
                    item_id = item.pop('id')
                    updates.append((item_id, item))
            
            # Only continue if we have valid updates
            if not updates:
                return error_response(404, f"No matching records found for update")
                
            logger.info(f"Processing {len(updates)} valid updates")
            result = self.db_service.bulk_update_records(model, updates)
            return success_response(200, f"Updated {len(result)} {model} records", result)
        else:
            # Single update operation (fallback for explicit record_id)
            if record_id is None:
                return error_response(400, "UPDATE operation requires 'id' field in data or separate 'id' parameter")
            
            if not data:
                return error_response(400, "UPDATE operation requires 'data'")
            
            # Handle custom ID field for single record update
            if id_field != 'id':
                logger.info(f"Looking for {model} record with {id_field}={record_id}")
                records = self.db_service.get_records(model, {id_field: record_id})
                logger.info(f"Found {len(records)} matching records")
                
                if records and len(records) > 0:
                    actual_id = records[0]['id']
                    logger.info(f"Using actual ID {actual_id} for record with {id_field}={record_id}")
                    result = self.db_service.update_record(model, actual_id, data)
                    if result:
                        return success_response(200, f"Updated {model} record", result)
                return error_response(404, f"{model} record with {id_field}={record_id} not found")
            else:
                # Direct ID update
                result = self.db_service.update_record(model, record_id, data)
                if result:
                    return success_response(200, f"Updated {model} record", result)
                return error_response(404, f"{model} record with ID {record_id} not found")


class DeleteHandler(BaseHandler):
    """Handler for DELETE operations"""
    
    def handle(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle DELETE requests
        
        Args:
            event (Dict[str, Any]): The event data
            
        Returns:
            Dict[str, Any]: Response object
        """
        model = event.get('model', '')
        data = event.get('data', {})
        record_id = event.get('id')
        id_field = event.get('id_field', 'id')
        
        # Validate parameters
        validation_error = self.validate_common_params(model, 'DELETE')
        if validation_error:
            return validation_error
        
        # For DELETE with custom id_field and no record_id, use data as the record_id
        if id_field != 'id' and record_id is None and not isinstance(data, (dict, list)):
            logger.info(f"Using data value as record identifier for field {id_field}")
            record_id = data
            data = {}  # Clear data to avoid confusion
        
        # Auto-detect batch mode
        is_batch = isinstance(data, list)
        batch_data = data if is_batch else []
        
        if is_batch:
            # Batch delete operation
            id_list = []
            id_field_list = []  # For custom ID fields
            
            for item in batch_data:
                if isinstance(item, dict):
                    # Extract ID from object using the specified id_field
                    if id_field in item:
                        id_field_list.append((id_field, item[id_field]))
                    elif 'id' in item:
                        id_list.append(item['id'])
                else:
                    # Direct value - could be either regular ID or custom field value
                    if id_field != 'id':
                        id_field_list.append((id_field, item))
                    else:
                        id_list.append(item)
            
            # If we have custom ID fields, use filtering to find and delete records
            if id_field != 'id' or id_field_list:
                total_deleted = 0
                all_deleted_records = []
                
                # Process items from id_field_list
                field_values = [(id_field, val) for _, val in id_field_list]
                
                # Also handle direct values in id_list if using custom id_field
                if id_field != 'id':
                    for val in id_list:
                        field_values.append((id_field, val))
                    id_list = []  # Clear id_list as we're handling these through field_values
                
                # Find and delete records for each field-value pair
                for field, value in field_values:
                    logger.info(f"Looking for {model} records with {field}={value}")
                    records = self.db_service.get_records(model, {field: value})
                    logger.info(f"Found {len(records)} matching records")
                    
                    for record in records:
                        if 'id' in record:
                            logger.info(f"Deleting record with ID {record['id']}")
                            if self.db_service.delete_record(model, record['id']):
                                total_deleted += 1
                                all_deleted_records.append(record)
                
                return success_response(200, f"Deleted {total_deleted} {model} records", {
                    "deleted_count": total_deleted,
                    "deleted_records": all_deleted_records
                })
            # Otherwise use regular bulk delete for direct IDs
            elif id_list:
                count = self.db_service.bulk_delete_records(model, id_list)
                return success_response(200, f"Deleted {count} {model} records", {"deleted_count": count})
            else:
                return error_response(400, "No valid IDs found for deletion")
        else:
            # Single record deletion
            if record_id is None:
                return error_response(400, "DELETE operation requires 'id'")
            
            # If using custom ID field, find by that field first
            if id_field != 'id':
                logger.info(f"Looking for {model} record with {id_field}={record_id}")
                records = self.db_service.get_records(model, {id_field: record_id})
                logger.info(f"Found {len(records)} matching records")
                
                if records:
                    # Delete first matching record
                    record = records[0]
                    logger.info(f"Deleting record with ID {record['id']}")
                    result = self.db_service.delete_record(model, record['id'])
                    if result:
                        return success_response(200, f"Deleted {model} record", {
                            "deleted": True,
                            "record": record
                        })
                return error_response(404, f"{model} record with {id_field}={record_id} not found")
            else:
                # Direct ID delete
                result = self.db_service.delete_record(model, record_id)
                if result:
                    return success_response(200, f"Deleted {model} record", {"deleted": True})
                return error_response(404, f"{model} record with ID {record_id} not found")


class RequestHandler:
    """Main request handler that routes to appropriate method handler"""
    
    def __init__(self):
        """Initialize the request handler with services and method handlers"""
        self.db_service = DatabaseOperationService()
        
        # Initialize method handlers
        self.handlers = {
            'POST': PostHandler(self.db_service),
            'GET': GetHandler(self.db_service),
            'UPDATE': UpdateHandler(self.db_service),
            'DELETE': DeleteHandler(self.db_service)
        }
    
    def handle_request(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming request by routing to the appropriate handler
        
        Args:
            event (Dict[str, Any]): The event data
            
        Returns:
            Dict[str, Any]: Response object
        """
        try:
            # Extract method and validate
            method = event.get('method', '').upper()
            
            if not method:
                return error_response(400, "Missing required parameter: 'method'")
            
            # Get the appropriate handler for the method
            handler = self.handlers.get(method)
            
            if not handler:
                return error_response(400, f"Unsupported method: {method}")
            
            # Log the request
            logger.info(f"Processing {method} request for model: {event.get('model', '')}")
            
            # Handle the request with the appropriate handler
            return handler.handle(event)
            
        except ValueError as e:
            logger.error(f"Value error: {str(e)}")
            return error_response(400, str(e))
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            return error_response(500, f"Internal server error: {str(e)}")


def success_response(status_code: int, message: str, data: Any = None) -> Dict[str, Any]:
    """
    Create a success response.
    
    Args:
        status_code (int): HTTP status code
        message (str): Success message
        data (any, optional): Response data
        
    Returns:
        dict: Response object
    """
    return {
        'statusCode': status_code,
        'body': {
            'success': True,
            'message': message,
            'data': data
        }
    }


def error_response(status_code: int, message: str) -> Dict[str, Any]:
    """
    Create an error response.
    
    Args:
        status_code (int): HTTP status code
        message (str): Error message
        
    Returns:
        dict: Response object
    """
    return {
        'statusCode': status_code,
        'body': {
            'success': False,
            'error': message
        }
    } 