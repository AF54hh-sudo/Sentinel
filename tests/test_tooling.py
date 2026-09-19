from sqlalchemy.exc import SQLAlchemyError

from scripts import audit_branding


def test_branding_audit_disposes_engine_when_postgresql_is_unavailable(monkeypatch):
    class OfflineEngine:
        disposed = False

        def dispose(self):
            self.disposed = True

    engine = OfflineEngine()
    monkeypatch.setattr(audit_branding, "create_database_engine", lambda database_url: engine)

    def unavailable_inspector(configured_engine):
        raise SQLAlchemyError("PostgreSQL unavailable")

    monkeypatch.setattr(audit_branding, "inspect", unavailable_inspector)

    assert audit_branding.database_matches() is None
    assert engine.disposed is True
