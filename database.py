import psycopg2
from psycopg2.extras import RealDictCursor
import face_recognition


class Database:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Database, cls).__new__(cls, *args, **kwargs)
            cls._instance.connection = None
        return cls._instance

    def connect(self):
        if self.connection is None:
            try:
                self.connection = psycopg2.connect(
                    dbname="face_recognition",
                    user="postgres",
                    password="Anchal@8084",
                    host="localhost",  # Change if using a remote database
                    port="5432"  # Default PostgreSQL port
                )
                print("Database connected successfully.")
            except Exception as e:
                print(f"Error connecting to database: {e}")

    def execute_query(self, query, params=None):
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if query.strip().upper().startswith("SELECT"):
                    return cursor.fetchall()
                self.connection.commit()
        except Exception as e:
            print(f"Error executing query: {e}")
            self.connection.rollback()

    def close_connection(self):
        if self.connection:
            self.connection.close()
            print("Database connection closed.")
            self.connection = None


if __name__ == "__main__":
    db = Database()
    db.connect()

    # Test query: Insert dummy data (optional for testing)
    test_query = "INSERT INTO faces (name, encoding) VALUES (%s, %s) RETURNING id;"
    dummy_encoding = b"dummy_face_encoding"  # Replace with actual encoding bytes later
    try:
        result = db.execute_query(test_query, ("Test User", dummy_encoding))
        print("Inserted ID:", result)
    except Exception as e:
        print("Error:", e)

    db.close_connection()