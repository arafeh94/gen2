import sqlite3
import argparse


class SQLiteDB:
    def __init__(self, db_name):
        """Initialize the database connection."""
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def execute_query(self, query, params=()):
        """Execute a single query."""
        self.cursor.execute(query, params)
        self.conn.commit()

    def execute_read_query(self, query, params=()):
        """Execute a read query and return the results."""
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def close(self):
        """Close the database connection."""
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="Interact with an SQLite database.")
    parser.add_argument("db_name", help="The name of the database file.")
    parser.add_argument("query", help="The SQL query to execute.")
    parser.add_argument("params", nargs="*", help="The parameters for the SQL query.", default=[])

    args = parser.parse_args()

    db = SQLiteDB(args.db_name)

    if args.query.lower().startswith("select"):
        results = db.execute_read_query(args.query, tuple(args.params))
        for row in results:
            print(row)
    else:
        db.execute_query(args.query, tuple(args.params))

    db.close()


if __name__ == "__main__":
    main()