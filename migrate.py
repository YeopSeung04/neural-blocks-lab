#!/usr/bin/env python3
import os
from pathlib import Path

from alembic import command
from alembic.config import Config

from backend import NeuralBlocksBackend
from mailer import MemoryMailer


ROOT = Path(__file__).resolve().parent
DATABASE_TARGET = os.environ.get(
    "NBL_DATABASE_URL",
    os.environ.get("NBL_DATABASE_PATH", ROOT / ".data" / "neural_blocks.db"),
)


def sqlalchemy_url(database_target):
    if database_target.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_target[len("postgresql://"):]
    if database_target.startswith("postgres://"):
        return "postgresql+psycopg://" + database_target[len("postgres://"):]
    return "sqlite:///" + str(Path(database_target).resolve())


def main():
    database_target = str(DATABASE_TARGET)
    NeuralBlocksBackend(
        database_target,
        mailer=MemoryMailer(),
        expose_dev_tokens=False,
    )
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", sqlalchemy_url(database_target))
    command.upgrade(config, "head")
    print("Database migrations are current.")


if __name__ == "__main__":
    main()
