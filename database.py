import re
import sqlite3
from pathlib import Path


class DatabaseConfigurationError(RuntimeError):
    pass


class DatabaseIntegrityError(RuntimeError):
    pass


class DatabaseConnection:
    def __init__(self, database, connection):
        self.database = database
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if error_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()
        return False

    def adapt_query(self, query, parameters):
        if self.database.engine == "sqlite":
            return query
        if query.strip().upper() == "BEGIN IMMEDIATE":
            return None
        if isinstance(parameters, dict):
            return re.sub(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)", r"%(\1)s", query)
        return query.replace("?", "%s")

    def execute(self, query, parameters=()):
        adapted = self.adapt_query(query, parameters)
        if adapted is None:
            return None
        try:
            return self.connection.execute(adapted, parameters)
        except self.database.integrity_error_types as error:
            raise DatabaseIntegrityError(str(error)) from error

    def executescript(self, script):
        if self.database.engine == "sqlite":
            return self.connection.executescript(script)
        for statement in script.split(";"):
            statement = statement.strip()
            if statement:
                self.execute(statement)
        return None

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()


class Database:
    def __init__(self, target):
        target = str(target)
        self.target = target
        self.engine = "postgres" if target.startswith(("postgres://", "postgresql://")) else "sqlite"
        self.integrity_error_types = (sqlite3.IntegrityError,)
        self.psycopg = None
        if self.engine == "sqlite":
            if target.startswith("sqlite:///"):
                target = target[10:]
            self.path = str(Path(target))
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.target = self.path
        else:
            try:
                import psycopg
                from psycopg import IntegrityError
            except ImportError as error:
                raise DatabaseConfigurationError(
                    "PostgreSQL requires psycopg. Install requirements.txt first."
                ) from error
            self.psycopg = psycopg
            self.integrity_error_types = (IntegrityError,)

    @property
    def description(self):
        return "postgresql" if self.engine == "postgres" else "sqlite"

    def connect(self):
        if self.engine == "sqlite":
            connection = sqlite3.connect(self.path, timeout=10)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 5000")
            return DatabaseConnection(self, connection)
        from psycopg.rows import dict_row

        connection = self.psycopg.connect(self.target, row_factory=dict_row)
        return DatabaseConnection(self, connection)

    def column_names(self, table):
        with self.connect() as db:
            if self.engine == "sqlite":
                rows = db.execute(f"PRAGMA table_info({table})").fetchall()
                return {row["name"] for row in rows}
            rows = db.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = ?
                """,
                (table,),
            ).fetchall()
            return {row["column_name"] for row in rows}

    def add_column_if_missing(self, table, column, definition):
        if column in self.column_names(table):
            return
        with self.connect() as db:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
