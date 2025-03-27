import sqlite3
import pandas as pd
import re

class DatabaseHandler:
    def __init__(self, db_name="temp_data.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    @staticmethod
    def _format_column_name(column):
        """
        Formats column names to snake_case:
        - Converts to lowercase
        - Replaces spaces and special characters (except underscores) with underscores
        - Ensures only single underscores are used
        - Preserves existing underscores
        """
        # Remove special characters except underscores and spaces
        cleaned = re.sub(r'[^a-zA-Z0-9_\s]', '', column)
        
        # Replace spaces with underscores
        with_underscores = re.sub(r'\s+', '_', cleaned)

        # Ensure only single underscores
        snake_case = re.sub(r'_+', '_', with_underscores).strip('_').lower()
        
        return snake_case
            

    def save_to_db(self, df, table_name):
        """
        Save DataFrame to database with LLM-friendly column names
        """
        # Create a copy of the DataFrame to avoid modifying the original
        formatted_df = df.copy()
        
        # Rename columns to LLM-friendly format
        formatted_df.columns = [self._format_column_name(col) for col in formatted_df.columns]
        
        # Save to SQL with formatted column names
        formatted_df.to_sql(table_name, self.conn, if_exists='replace', index=False)
        
        # Commit the changes
        self.conn.commit()

    def query_db(self, query):
        """
        Execute a SQL query and return results
        """
        return pd.read_sql_query(query, self.conn)

    def close_connection(self):
        """
        Close the database connection
        """
        self.conn.close()

    def display_table_info(self, table_name):
        """
        Display information about the table columns
        """
        query = f"PRAGMA table_info({table_name})"
        column_info = pd.read_sql_query(query, self.conn)
        print(f"Columns in {table_name}:")
        for _, row in column_info.iterrows():
            print(f"- {row['name']}")