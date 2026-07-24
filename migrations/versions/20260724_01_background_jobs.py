"""Add persistent background jobs.

Revision ID: 20260724_01
Revises:
Create Date: 2026-07-24
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260724_01"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if "background_jobs" not in inspector.get_table_names():
        op.create_table(
            "background_jobs",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("tenant_id", sa.Text(), nullable=True),
            sa.Column("user_id", sa.Text(), nullable=True),
            sa.Column("job_type", sa.Text(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("dedupe_key", sa.Text(), nullable=True),
            sa.Column("payload_text", sa.Text(), nullable=False),
            sa.Column(
                "payload_encrypted",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            sa.Column("result_json", sa.Text(), nullable=True),
            sa.Column("error_code", sa.Text(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
            sa.Column(
                "max_attempts",
                sa.Integer(),
                nullable=False,
                server_default="3",
            ),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
            sa.Column("started_at", sa.Text(), nullable=True),
            sa.Column("finished_at", sa.Text(), nullable=True),
        )
    existing_indexes = {
        item["name"]
        for item in sa.inspect(connection).get_indexes("background_jobs")
    }
    for name, columns in (
        ("idx_background_jobs_tenant", ["tenant_id", "created_at"]),
        ("idx_background_jobs_status", ["status", "updated_at"]),
        ("idx_background_jobs_dedupe", ["dedupe_key", "status"]),
    ):
        if name not in existing_indexes:
            op.create_index(name, "background_jobs", columns)


def downgrade() -> None:
    connection = op.get_bind()
    if "background_jobs" in sa.inspect(connection).get_table_names():
        op.drop_table("background_jobs")
