from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create a database file on disk
engine = create_engine('sqlite:///chess_assistant.db', echo=False)

# Base class that all our table models will inherit from
Base = declarative_base()

# Session factory to talk to the database
SessionLocal = sessionmaker(bind=engine)

def get_session():
    # Creates a new database session
    return SessionLocal()

def init_db():
    # Creates all tables in the database.
    Base.metadata.create_all(bind = engine)

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")
    print(f"Database file created at: chess_assistant.db")
