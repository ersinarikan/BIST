"""
add user role and last_login_ip columns

Revision ID: 20250821_add_user_role_last_login_ip
Revises: 
Create Date: 2025-08-21 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a20250821_user_role_ip'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Add 'role' column with default 'user' and index
    op.add_column('users', sa.Column('role', sa.String(length=20), nullable=False, server_default='user'))
    op.create_index('ix_users_role', 'users', ['role'], unique=False)

    # Add 'last_login_ip' column
    op.add_column('users', sa.Column('last_login_ip', sa.String(length=45), nullable=True))

    # Optional: remove server_default for 'role' after backfilling existing rows
    with op.batch_alter_table('users') as batch_op:
        batch_op.alter_column('role', server_default=None)


def downgrade():
    # Drop columns and index
    with op.batch_alter_table('users') as batch_op:
        batch_op.drop_column('last_login_ip')
        batch_op.drop_column('role')
    op.drop_index('ix_users_role', table_name='users')


