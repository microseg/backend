import sys
import logging
import psycopg2
import os
from typing import List, Tuple

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class DatabaseConnector:
    """
    Database connection manager implementing the Singleton pattern.
    Handles database connections and query execution.
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls, *args, **kwargs):
        """
        Ensure only one instance of DatabaseConnector exists.
        
        Returns:
            DatabaseConnector: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(DatabaseConnector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize database connection parameters from environment variables.
        Connection is only initialized once due to singleton pattern.
        """
        if self._initialized:
            return
            
        # Get database configuration from environment variables
        self.db_host = os.environ['DB_HOST']
        self.db_port = int(os.environ.get('DB_PORT', 5432))
        self.db_name = os.environ['DB_NAME']
        self.db_user = os.environ['DB_USER']
        self.db_password = os.environ['DB_PASSWORD']
        
        self.conn = None
        self._connect()
        self._initialized = True
    
    def _connect(self):
        """
        Establish database connection.
        
        Raises:
            ConnectionError: If connection to database fails
        """
        try:
            self.conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
                connect_timeout=5
            )
            logger.info("✅ SUCCESS: Connected to PostgreSQL RDS instance.")
        except psycopg2.Error as e:
            logger.error("ERROR: Could not connect to PostgreSQL RDS instance.")
            logger.error(e)
            raise ConnectionError(f"Failed to connect to database: {str(e)}")
    
    def get_connection(self):
        """
        Get the database connection, establishing a new one if necessary.
        
        Returns:
            connection: PostgreSQL database connection
        """
        if not self.conn or self.conn.closed:
            self._connect()
        return self.conn
    
    def execute_query(self, query: str, params: tuple = None) -> List[tuple]:
        """
        Execute a database query and return results.
        
        Args:
            query (str): SQL query to execute
            params (tuple, optional): Query parameters
            
        Returns:
            List[tuple]: Query results as list of tuples
        """
        conn = self.get_connection()
        results = []
        
        try:
            with conn.cursor() as cur:
                # Log the query for debugging
                logger.info(f"Executing query: {query}")
                logger.info(f"With parameters: {params}")
                
                # Execute the query
                cur.execute(query, params)
                
                # If the query returns results, fetch them
                if cur.description:
                    results = cur.fetchall()
                    logger.info(f"Query returned {len(results)} rows")
                
                # Check if this is a write operation (INSERT, UPDATE, DELETE)
                is_write_operation = (
                    query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')) or
                    'CREATE TABLE' in query.upper()
                )
                
                # Commit the transaction for write operations
                if is_write_operation:
                    conn.commit()
                    logger.info("Transaction committed")
        except Exception as e:
            # Log detailed error information
            logger.error(f"Database error executing query: {str(e)}")
            logger.error(f"Query was: {query}")
            logger.error(f"Params were: {params}")
            
            # Rollback the transaction on error
            try:
                conn.rollback()
                logger.info("Transaction rolled back due to error")
            except Exception as rollback_error:
                logger.error(f"Error during rollback: {str(rollback_error)}")
            
            # Re-raise the exception
            raise
        
        return results
    
    def execute_transaction(self, queries: List[Tuple[str, tuple]]) -> None:
        """
        Execute multiple queries as a single transaction.
        
        Args:
            queries (List[Tuple[str, tuple]]): List of (query, params) tuples
        """
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                # Log transaction information
                logger.info(f"Starting transaction with {len(queries)} queries")
                
                # Execute each query in the transaction
                for i, (query, params) in enumerate(queries):
                    logger.info(f"Executing query {i+1}/{len(queries)}: {query}")
                    logger.info(f"With parameters: {params}")
                    
                    cur.execute(query, params)
                    
                    # If query returns results, log them
                    if cur.description:
                        results = cur.fetchall()
                        logger.info(f"Query {i+1} returned {len(results)} rows")
                
                # Commit the transaction
                conn.commit()
                logger.info("Transaction committed successfully")
        except Exception as e:
            # Log detailed error information
            logger.error(f"Transaction failed: {str(e)}")
            
            # Rollback the transaction on error
            try:
                conn.rollback()
                logger.info("Transaction rolled back due to error")
            except Exception as rollback_error:
                logger.error(f"Error during rollback: {str(rollback_error)}")
            
            # Re-raise the exception
            raise 