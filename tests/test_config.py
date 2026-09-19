import logging

from sentinel.config.settings import PROJECT_ROOT, Settings
from sentinel.logging import configure_logging


def test_settings_env_file_is_project_root_absolute_path():
    env_file = Settings.model_config["env_file"]
    assert env_file == PROJECT_ROOT / ".env"
    assert env_file.is_absolute()


def test_phase14_embedding_defaults_are_explicit():
    settings = Settings(_env_file=None)
    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.documents_dir.as_posix() == "data/documents"


def test_logging_configuration_accepts_known_levels_and_falls_back_to_info():
    configure_logging("debug")
    assert logging.getLogger().level == logging.DEBUG

    configure_logging("not-a-level")
    assert logging.getLogger().level == logging.INFO
