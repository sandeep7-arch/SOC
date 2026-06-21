from contextlib import contextmanager
from base import SessionLocal

@contextmanager
def get_db_session():
    session = SessionLocal() # Creates a session
    try: # means attempt the code
        yield session # Pause and hands session to caller
        session.commit() # Saves the changes permanently to the database file
    except Exception: # if risky code crashes, run below block, instead of xrashing the whole program
        session.rollback() # undoes any changes made during the last commit, so that if there is a crash, it won't save it
        raise # tells the user that something went wrong
    finally:    # runs no matter the risky code runs or crashes
        session.close() 

if __name__ == "__main__":
    from base import init_db

    init_db()

    with get_db_session() as session: # Uses a session that was yielded, and after the below block runs it resumes after yield
        print("Session created successfully:", session)
        print("Session is active:", session.is_active) # A true/false check, that tells if the session is currently usable or closed