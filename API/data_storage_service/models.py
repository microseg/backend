import logging
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from db_connector import DatabaseConnector

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class IdGenerator:
    """
    ID generator class that provides common ID generation functionality.
    """
    
    def __init__(self, db_connector: DatabaseConnector):
        """
        Initialize the ID generator.
        
        Args:
            db_connector (DatabaseConnector): Database connector
        """
        self.db = db_connector
        
    def generate_id(self, table_name: str) -> int:
        """
        Generate a new ID based on the current state of the database.
        Gets the maximum ID from the specified table and increments it by 1.
        If the table is empty, returns 1 as the first ID.
        
        Args:
            table_name (str): Table name
            
        Returns:
            int: A new unique ID
        """
        query = f"SELECT MAX(id) FROM {table_name};"
        rows = self.db.execute_query(query)
        
        # Log the result for debugging
        logger.info(f"Table {table_name} max ID query result: {rows}")
        
        # If no rows or NULL result, return 1 as the first ID
        if not rows or rows[0][0] is None:
            return 1
        
        # Return max ID + 1
        return rows[0][0] + 1
        
    def generate_batch_ids(self, table_name: str, count: int) -> List[int]:
        """
        Generate a consecutive list of IDs for batch operations.
        
        Args:
            table_name (str): Table name
            count (int): Number of IDs needed
            
        Returns:
            List[int]: List of generated IDs
        """
        # Get the current max ID
        start_id = self.generate_id(table_name) - 1  # Subtract 1 because generate_id already adds 1
        
        # Generate a consecutive list of IDs
        return [start_id + i + 1 for i in range(count)]


class BaseModel(ABC):
    """
    Abstract base class for all database models.
    Defines common interface and functionality.
    """
    
    def __init__(self):
        """
        Initialize model with database connector.
        """
        self.db = DatabaseConnector()
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """
        Get the database table name.
        
        Returns:
            str: Table name
        """
        pass
    
    @property
    @abstractmethod
    def create_table_query(self) -> str:
        """
        Get the SQL query for table creation.
        
        Returns:
            str: SQL CREATE TABLE query
        """
        pass
    
    @abstractmethod
    def get_insert_query(self, data: Dict[str, Any]) -> Tuple[str, tuple]:
        """
        Get the SQL query and parameters for inserting records.
        
        Args:
            data (Dict[str, Any]): Data to insert
            
        Returns:
            Tuple[str, tuple]: (SQL query, parameters)
        """
        pass
    
    @abstractmethod
    def get_update_query(self, id_value: Any, data: Dict[str, Any]) -> Tuple[str, tuple]:
        """
        Get the SQL query and parameters for updating records.
        
        Args:
            id_value (Any): Primary key value to identify the record
            data (Dict[str, Any]): Data to update
            
        Returns:
            Tuple[str, tuple]: (SQL query, parameters)
        """
        pass
    
    @abstractmethod
    def get_delete_query(self, id_value: Any) -> Tuple[str, tuple]:
        """
        Get the SQL query and parameters for deleting records.
        
        Args:
            id_value (Any): Primary key value to identify the record
            
        Returns:
            Tuple[str, tuple]: (SQL query, parameters)
        """
        pass
    
    @abstractmethod
    def map_row_to_dict(self, row: tuple) -> Dict[str, Any]:
        """
        Map a database row to a dictionary.
        
        Args:
            row (tuple): Database row
            
        Returns:
            Dict[str, Any]: Dictionary representation of the row
        """
        pass
    
    @property
    @abstractmethod
    def id_column(self) -> str:
        """
        Get the primary key column name.
        
        Returns:
            str: Primary key column name
        """
        pass
    
    def create_table_if_not_exists(self) -> None:
        """
        Create the table if it doesn't exist.
        """
        self.db.execute_query(self.create_table_query)
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all records from the table.
        
        Returns:
            List[Dict[str, Any]]: List of records as dictionaries
        """
        query = f"SELECT * FROM {self.table_name};"
        rows = self.db.execute_query(query)
        return [self.map_row_to_dict(row) for row in rows]
    
    def get_by_id(self, id_value: Any) -> Optional[Dict[str, Any]]:
        """
        Get a record by its ID.
        
        Args:
            id_value (Any): Primary key value
            
        Returns:
            Optional[Dict[str, Any]]: Record as dictionary or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE {self.id_column} = %s;"
        rows = self.db.execute_query(query, (id_value,))
        if not rows:
            return None
        return self.map_row_to_dict(rows[0])
    
    def get_by_filter(self, filter_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get records by filter criteria.
        
        Args:
            filter_dict (Dict[str, Any]): Filter criteria as field-value pairs
            
        Returns:
            List[Dict[str, Any]]: List of matching records
        """
        if not filter_dict:
            return self.get_all()
        
        where_clauses = []
        params = []
        
        # Log the filter operation
        logger.info(f"Filtering {self.table_name} with criteria: {filter_dict}")
        
        for key, value in filter_dict.items():
            # Check if value is a list/array for IN clause
            if isinstance(value, (list, tuple)):
                if value:  # Only process non-empty lists
                    # Create placeholders for IN clause (%s, %s, %s, ...)
                    placeholders = ', '.join(['%s'] * len(value))
                    where_clauses.append(f"{key} IN ({placeholders})")
                    # Add each value in the list to params
                    params.extend(value)
                    logger.info(f"Adding IN filter: {key} IN {value} (type: {type(value).__name__})")
            else:
                # Normal equality filter
                where_clauses.append(f"{key} = %s")
                params.append(value)
                logger.info(f"Adding filter: {key} = {value} (type: {type(value).__name__})")
        
        query = f"SELECT * FROM {self.table_name} WHERE {' AND '.join(where_clauses)};"
        logger.info(f"Executing filter query: {query} with params: {params}")
        
        rows = self.db.execute_query(query, params)
        logger.info(f"Filter query returned {len(rows)} rows")
        
        result = [self.map_row_to_dict(row) for row in rows]
        return result
    
    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new record.
        
        Args:
            data (Dict[str, Any]): Data to insert
            
        Returns:
            Dict[str, Any]: Created record
        """
        # Ensure table exists
        self.create_table_if_not_exists()
        
        # Get insert query and execute
        query, params = self.get_insert_query(data)
        
        # Execute query and get result
        result_rows = self.db.execute_query(query, params)
        
        # Log the operation for debugging
        logger.info(f"Executed INSERT query with params: {params}")
        logger.info(f"Query result: {result_rows}")
        
        # If we got a result with an ID, fetch the record by that ID
        if result_rows and len(result_rows) > 0 and len(result_rows[0]) > 0:
            inserted_id = result_rows[0][0]  # First column of first row should be the ID
            logger.info(f"Record inserted with ID: {inserted_id}")
            return self.get_by_id(inserted_id)
        
        # If we didn't get a result with an ID, try to fetch by unique fields
        # Create a filter with only the fields that should be unique
        unique_filter = {}
        if hasattr(self, 'unique_fields'):
            for field in self.unique_fields:
                if field in data:
                    unique_filter[field] = data[field]
        
        # If we have unique fields, try to fetch by them
        if unique_filter:
            fetched_records = self.get_by_filter(unique_filter)
            if fetched_records and len(fetched_records) > 0:
                return fetched_records[0]
        
        # As a last resort, return the input data
        # This is not ideal but it's better than nothing
        logger.warning("Could not confirm record insertion, returning input data")
        return data
    
    def update(self, id_value: Any, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an existing record.
        
        Args:
            id_value (Any): Primary key value
            data (Dict[str, Any]): Data to update
            
        Returns:
            Optional[Dict[str, Any]]: Updated record or None if not found
        """
        # Check if record exists
        if not self.get_by_id(id_value):
            return None
        
        # Get update query and execute
        query, params = self.get_update_query(id_value, data)
        self.db.execute_query(query, params)
        
        # Return updated record
        return self.get_by_id(id_value)
    
    def delete(self, id_value: Any) -> bool:
        """
        Delete a record.
        
        Args:
            id_value (Any): Primary key value
            
        Returns:
            bool: True if deleted, False if not found
        """
        # Check if record exists
        if not self.get_by_id(id_value):
            return False
        
        # Get delete query and execute
        query, params = self.get_delete_query(id_value)
        self.db.execute_query(query, params)
        
        return True
    
    def bulk_create(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create multiple records in a transaction.
        
        Args:
            data_list (List[Dict[str, Any]]): List of data to insert
            
        Returns:
            List[Dict[str, Any]]: List of created records
        """
        # Ensure table exists
        self.create_table_if_not_exists()
        
        # Prepare transaction queries
        transaction_queries = [self.get_insert_query(data) for data in data_list]
        
        # Execute transaction
        self.db.execute_transaction(transaction_queries)
        
        # Return simplified response (actual retrieval might be more complex)
        return data_list
    
    def bulk_update(self, updates: List[Tuple[Any, Dict[str, Any]]]) -> List[Optional[Dict[str, Any]]]:
        """
        Update multiple records in a transaction.
        
        Args:
            updates (List[Tuple[Any, Dict[str, Any]]]): List of (id_value, data) tuples
            
        Returns:
            List[Optional[Dict[str, Any]]]: List of updated records
        """
        # Prepare transaction queries
        transaction_queries = []
        valid_ids = []
        
        for id_value, data in updates:
            if self.get_by_id(id_value):
                transaction_queries.append(self.get_update_query(id_value, data))
                valid_ids.append(id_value)
        
        # Execute transaction
        if transaction_queries:
            self.db.execute_transaction(transaction_queries)
        
        # Return updated records
        return [self.get_by_id(id_value) for id_value in valid_ids]
    
    def bulk_delete(self, id_values: List[Any]) -> int:
        """
        Delete multiple records in a transaction.
        
        Args:
            id_values (List[Any]): List of primary key values
            
        Returns:
            int: Number of records deleted
        """
        # Prepare transaction queries
        transaction_queries = []
        valid_ids = []
        
        for id_value in id_values:
            if self.get_by_id(id_value):
                transaction_queries.append(self.get_delete_query(id_value))
                valid_ids.append(id_value)
        
        # Execute transaction
        if transaction_queries:
            self.db.execute_transaction(transaction_queries)
        
        return len(valid_ids)


class UserInfo(BaseModel):
    """
    User information model for managing user data from Cognito.
    """
    
    # Add unique fields property
    unique_fields = ['cognito_sub', 'email']
    
    @property
    def table_name(self) -> str:
        """
        Get the users table name.
        
        Returns:
            str: Table name
        """
        return "users"
    
    @property
    def id_column(self) -> str:
        """
        Get the primary key column name.
        
        Returns:
            str: Primary key column name
        """
        return "id"
    
    @property
    def create_table_query(self) -> str:
        """
        Get the SQL query for creating users table.
        
        Returns:
            str: SQL CREATE TABLE query
        """
        return """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                cognito_sub VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) NOT NULL,
                username VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
    
    def get_insert_query(self, data: Dict[str, Any]) -> Tuple[str, tuple]:
        """
        Get the SQL query and parameters for inserting a user.
        Uses UPSERT pattern (INSERT with ON CONFLICT DO UPDATE).
        
        Args:
            data (Dict[str, Any]): User data
            
        Returns:
            Tuple[str, tuple]: (SQL query, parameters)
        """
        # Build lists of columns and placeholders for the SQL query
        columns = []
        placeholders = []
        values = []
        
        # Add each field to the query
        for field, value in data.items():
            # Skip id if it's auto-generated (let the database assign it)
            if field == 'id' and not value:
                continue
                
            columns.append(field)
            placeholders.append('%s')
            values.append(value)
        
        # Build the query
        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT (cognito_sub)
            DO UPDATE SET
                email = EXCLUDED.email,
                username = EXCLUDED.username,
                created_at = EXCLUDED.created_at
            RETURNING id;
        """
        
        # Log the query for debugging
        logger.info(f"Generated insert query: {query}")
        logger.info(f"With values: {values}")
        
        return (query, tuple(values))
    
    def get_update_query(self, id_value: Any, data: Dict[str, Any]) -> Tuple[str, tuple]:
        """
        Get the SQL query and parameters for updating a user.
        
        Args:
            id_value (Any): User ID
            data (Dict[str, Any]): User data to update
            
        Returns:
            Tuple[str, tuple]: (SQL query, parameters)
        """
        set_clauses = []
        params = []
        
        # Only update fields that are provided
        if 'email' in data:
            set_clauses.append("email = %s")
            params.append(data['email'])
        
        if 'username' in data:
            set_clauses.append("username = %s")
            params.append(data['username'])
        
        # Add more fields as needed
        
        # Add the ID parameter
        params.append(id_value)
        
        query = f"""
            UPDATE users 
            SET {', '.join(set_clauses)}
            WHERE id = %s
            RETURNING *;
        """
        
        return (query, tuple(params))
    
    def get_delete_query(self, id_value: Any) -> Tuple[str, tuple]:
        """
        Get the SQL query and parameters for deleting a user.
        
        Args:
            id_value (Any): User ID
            
        Returns:
            Tuple[str, tuple]: (SQL query, parameters)
        """
        query = """
            DELETE FROM users
            WHERE id = %s;
        """
        return (query, (id_value,))
    
    def map_row_to_dict(self, row: tuple) -> Dict[str, Any]:
        """
        Map a users table row to a dictionary.
        
        Args:
            row (tuple): Database row
            
        Returns:
            Dict[str, Any]: Dictionary representation of the user
        """
        return {
            "id": row[0],
            "cognito_sub": row[1],
            "email": row[2],
            "username": row[3],
            "created_at": row[4].isoformat() if isinstance(row[4], datetime) else row[4]
        }
    
    def generate_id(self) -> int:
        """
        Generate a new ID for a user based on the current state of the database.
        This will get the max ID from the users table and increment it by 1.
        If the table is empty, it returns 1 as the first ID.
        
        Returns:
            int: A new unique ID for a user
        """
        id_generator = IdGenerator(self.db)
        return id_generator.generate_id(self.table_name)
    
    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new user record with auto-generated ID if not provided.
        
        Args:
            data (Dict[str, Any]): User data to insert
            
        Returns:
            Dict[str, Any]: Created user record
        """
        # Make a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Auto-generate ID if not provided or is None/empty
        if 'id' not in data_copy or not data_copy['id']:
            data_copy['id'] = self.generate_id()
            logger.info(f"Auto-generated ID for new user: {data_copy['id']}")
        
        # Call the parent class create method with the updated data
        return super().create(data_copy)
    
    def bulk_create(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create multiple user records with auto-generated IDs if not provided.
        
        Args:
            data_list (List[Dict[str, Any]]): List of user data to insert
            
        Returns:
            List[Dict[str, Any]]: List of created user records
        """
        # Make copies to avoid modifying the original data
        data_copies = []
        
        # Use ID generator to generate all needed IDs at once
        id_generator = IdGenerator(self.db)
        # Calculate how many IDs we need to generate (only for records without IDs)
        need_ids_count = sum(1 for data in data_list if 'id' not in data or not data['id'])
        
        if need_ids_count > 0:
            # Get a batch of consecutive IDs
            ids = id_generator.generate_batch_ids(self.table_name, need_ids_count)
            id_index = 0  # Track the number of IDs used
            
            for data in data_list:
                data_copy = data.copy()
                
                # Assign IDs to records that don't have one
                if 'id' not in data_copy or not data_copy['id']:
                    data_copy['id'] = ids[id_index]
                    logger.info(f"Auto-generated ID for batch user: {data_copy['id']}")
                    id_index += 1
                
                data_copies.append(data_copy)
        else:
            # All records already have IDs, just copy them
            data_copies = [data.copy() for data in data_list]
        
        # Call the parent class bulk_create method with the updated data
        return super().bulk_create(data_copies)


class ImageInfo(BaseModel):
    """
    Image information model for managing uploaded images.
    """
    
    @property
    def table_name(self) -> str:
        """
        Get the images table name.
        
        Returns:
            str: Table name
        """
        return "images"
    
    @property
    def id_column(self) -> str:
        """
        Get the primary key column name.
        
        Returns:
            str: Primary key column name
        """
        return "id"
    
    @property
    def create_table_query(self) -> str:
        """
        Get the SQL query for creating images table.
        
        Returns:
            str: SQL CREATE TABLE query
        """
        return """
            CREATE TABLE IF NOT EXISTS images (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                s3_key VARCHAR(255) NOT NULL,
                original_filename VARCHAR(255) NOT NULL,
                content_type VARCHAR(100) NOT NULL,
                file_size BIGINT NOT NULL,
                width INTEGER,
                height INTEGER,
                upload_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                folder_path VARCHAR(255) DEFAULT '',
                description TEXT,
                is_deleted BOOLEAN DEFAULT FALSE,
                deleted_at TIMESTAMP WITH TIME ZONE,
                UNIQUE(user_id, s3_key)
            );
        """
    
    def get_insert_query(self, data: Dict[str, Any]) -> Tuple[str, tuple]:
        """
        Get the SQL query and parameters for inserting an image.
        
        Args:
            data (Dict[str, Any]): Image data
            
        Returns:
            Tuple[str, tuple]: (SQL query, parameters)
        """
        # Build lists of columns and placeholders for the SQL query
        columns = []
        placeholders = []
        values = []
        
        # Add each field to the query
        for field, value in data.items():
            # Skip id if it's auto-generated (let the database assign it)
            if field == 'id' and not value:
                continue
                
            columns.append(field)
            placeholders.append('%s')
            values.append(value)
        
        # Build the query
        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING id;
        """
        
        # Log the query for debugging
        logger.info(f"Generated image insert query: {query}")
        logger.info(f"With values: {values}")
        
        return (query, tuple(values))
    
    def get_update_query(self, id_value: Any, data: Dict[str, Any]) -> Tuple[str, tuple]:
        """
        Get the SQL query and parameters for updating an image.
        
        Args:
            id_value (Any): Image ID
            data (Dict[str, Any]): Image data to update
            
        Returns:
            Tuple[str, tuple]: (SQL query, parameters)
        """
        set_clauses = []
        params = []
        
        # Only update fields that are provided
        for field, value in data.items():
            # Skip id field as it can't be updated
            if field == 'id':
                continue
                
            set_clauses.append(f"{field} = %s")
            params.append(value)
        
        # Add the ID parameter
        params.append(id_value)
        
        query = f"""
            UPDATE {self.table_name} 
            SET {', '.join(set_clauses)}
            WHERE id = %s
            RETURNING *;
        """
        
        return (query, tuple(params))
    
    def get_delete_query(self, id_value: Any) -> Tuple[str, tuple]:
        """
        Get the SQL query and parameters for deleting an image.
        
        Args:
            id_value (Any): Image ID
            
        Returns:
            Tuple[str, tuple]: (SQL query, parameters)
        """
        query = f"""
            DELETE FROM {self.table_name}
            WHERE id = %s;
        """
        return (query, (id_value,))
    
    def map_row_to_dict(self, row: tuple) -> Dict[str, Any]:
        """
        Map an images table row to a dictionary.
        
        Args:
            row (tuple): Database row
            
        Returns:
            Dict[str, Any]: Dictionary representation of the image
        """
        # Convert timestamp fields to ISO format strings
        upload_time = row[8].isoformat() if isinstance(row[8], datetime) else row[8]
        deleted_at = row[12].isoformat() if row[12] is not None and isinstance(row[12], datetime) else row[12]
        
        return {
            "id": row[0],
            "user_id": row[1],
            "s3_key": row[2],
            "original_filename": row[3],
            "content_type": row[4],
            "file_size": row[5],
            "width": row[6],
            "height": row[7],
            "upload_time": upload_time,
            "folder_path": row[9],
            "description": row[10],
            "is_deleted": row[11],
            "deleted_at": deleted_at
        }

    def generate_id(self) -> int:
        """
        Generate a new ID for an image based on the current state of the database.
        This will get the max ID from the images table and increment it by 1.
        If the table is empty, it returns 1 as the first ID.
        
        Returns:
            int: A new unique ID for an image
        """
        id_generator = IdGenerator(self.db)
        return id_generator.generate_id(self.table_name)
    
    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new image record with auto-generated ID if not provided.
        
        Args:
            data (Dict[str, Any]): Image data to insert
            
        Returns:
            Dict[str, Any]: Created image record
        """
        # Make a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Auto-generate ID if not provided or is None/empty
        if 'id' not in data_copy or not data_copy['id']:
            data_copy['id'] = self.generate_id()
            logger.info(f"Auto-generated ID for new image: {data_copy['id']}")
        
        # Call the parent class create method with the updated data
        return super().create(data_copy)
    
    def bulk_create(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create multiple image records with auto-generated IDs if not provided.
        
        Args:
            data_list (List[Dict[str, Any]]): List of image data to insert
            
        Returns:
            List[Dict[str, Any]]: List of created image records
        """
        # Make copies to avoid modifying the original data
        data_copies = []
        
        # Use ID generator to generate all needed IDs at once
        id_generator = IdGenerator(self.db)
        # Calculate how many IDs we need to generate (only for records without IDs)
        need_ids_count = sum(1 for data in data_list if 'id' not in data or not data['id'])
        
        if need_ids_count > 0:
            # Get a batch of consecutive IDs
            ids = id_generator.generate_batch_ids(self.table_name, need_ids_count)
            id_index = 0  # Track the number of IDs used
            
            for data in data_list:
                data_copy = data.copy()
                
                # Assign IDs to records that don't have one
                if 'id' not in data_copy or not data_copy['id']:
                    data_copy['id'] = ids[id_index]
                    logger.info(f"Auto-generated ID for batch image: {data_copy['id']}")
                    id_index += 1
                
                data_copies.append(data_copy)
        else:
            # All records already have IDs, just copy them
            data_copies = [data.copy() for data in data_list]
        
        # Call the parent class bulk_create method with the updated data
        return super().bulk_create(data_copies)


class DatabaseOperationService:
    """
    Generic service for database operations across different models.
    Provides standardized methods for CRUD operations.
    """
    
    def __init__(self):
        """
        Initialize service with available models.
        New models should be added here.
        """
        self.models = {
            'users': UserInfo(),
            'images': ImageInfo()
            # Add more models here as they are created
            # 'products': ProductInfo(),
            # 'orders': OrderInfo(),
            # etc.
        }
    
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """
        Get a model by name.
        
        Args:
            model_name (str): Name of the model (table)
            
        Returns:
            Optional[BaseModel]: Model instance or None if not found
        """
        return self.models.get(model_name.lower())
    
    def create_record(self, model_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new record in the specified model.
        
        Args:
            model_name (str): Name of the model (table)
            data (Dict[str, Any]): Data to insert
            
        Returns:
            Dict[str, Any]: Created record
            
        Raises:
            ValueError: If model not found
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        return model.create(data)
    
    def get_records(self, model_name: str, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get records from the specified model.
        
        Args:
            model_name (str): Name of the model (table)
            filter_dict (Dict[str, Any], optional): Filter criteria
            
        Returns:
            List[Dict[str, Any]]: List of records
            
        Raises:
            ValueError: If model not found
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        if filter_dict:
            return model.get_by_filter(filter_dict)
        return model.get_all()
    
    def get_record_by_id(self, model_name: str, id_value: Any) -> Optional[Dict[str, Any]]:
        """
        Get a record by ID from the specified model.
        
        Args:
            model_name (str): Name of the model (table)
            id_value (Any): Primary key value
            
        Returns:
            Optional[Dict[str, Any]]: Record or None if not found
            
        Raises:
            ValueError: If model not found
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        return model.get_by_id(id_value)
    
    def update_record(self, model_name: str, id_value: Any, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a record in the specified model.
        
        Args:
            model_name (str): Name of the model (table)
            id_value (Any): Primary key value
            data (Dict[str, Any]): Data to update
            
        Returns:
            Optional[Dict[str, Any]]: Updated record or None if not found
            
        Raises:
            ValueError: If model not found
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        return model.update(id_value, data)
    
    def delete_record(self, model_name: str, id_value: Any) -> bool:
        """
        Delete a record from the specified model.
        
        Args:
            model_name (str): Name of the model (table)
            id_value (Any): Primary key value
            
        Returns:
            bool: True if deleted, False if not found
            
        Raises:
            ValueError: If model not found
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        return model.delete(id_value)
    
    def bulk_create_records(self, model_name: str, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create multiple records in the specified model.
        
        Args:
            model_name (str): Name of the model (table)
            data_list (List[Dict[str, Any]]): List of data to insert
            
        Returns:
            List[Dict[str, Any]]: List of created records
            
        Raises:
            ValueError: If model not found
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        return model.bulk_create(data_list)
    
    def bulk_update_records(self, model_name: str, updates: List[Tuple[Any, Dict[str, Any]]]) -> List[Optional[Dict[str, Any]]]:
        """
        Update multiple records in the specified model.
        
        Args:
            model_name (str): Name of the model (table)
            updates (List[Tuple[Any, Dict[str, Any]]]): List of (id_value, data) tuples
            
        Returns:
            List[Optional[Dict[str, Any]]]: List of updated records
            
        Raises:
            ValueError: If model not found
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        return model.bulk_update(updates)
    
    def bulk_delete_records(self, model_name: str, id_values: List[Any]) -> int:
        """
        Delete multiple records in the specified model.
        
        Args:
            model_name (str): Name of the model (table)
            id_values (List[Any]): List of primary key values
            
        Returns:
            int: Number of records deleted
            
        Raises:
            ValueError: If model not found
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        return model.bulk_delete(id_values) 