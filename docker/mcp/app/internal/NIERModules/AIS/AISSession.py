from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import InvalidRequestError


class ReadOnlySession(sessionmaker):
    """Session factory that disables mutating operations."""

    def commit(self):
        raise InvalidRequestError("This session is read-only. Commit is disabled.")

    def delete(self, instance):
        raise InvalidRequestError("This session is read-only. Delete is disabled.")

    def add(self, instance):
        raise InvalidRequestError("This session is read-only. Add is disabled.")
