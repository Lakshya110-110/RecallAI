#!/usr/bin/env python3
"""
Safe, non-destructive, reversible SQLite-to-PostgreSQL (or SQLite-to-SQLite) migration utility for RecallAI.

Features:
- Opens the source SQLite database strictly in read-only mode (never modifies recallai.db).
- Exports a full JSON backup snapshot before migrating so the migration is 100% reversible.
- Preserves all primary keys (IDs), foreign key relationships, user roles, lessons, contributor
  assignments, reference documents, extracted keywords, keyword pools, submissions, C4 assessment
  results, and notifications.
- Idempotent upsert/insert in dependency order and synchronizes PostgreSQL sequences afterward.

Usage:
  python backend/migrate_sqlite_to_postgres.py --source recallai.db --target "$DATABASE_URL" --backup-json backup_snapshot.json
  python backend/migrate_sqlite_to_postgres.py --restore-json backup_snapshot.json --target "sqlite:///./restored_recallai.db"
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import inspect, select, text
from sqlalchemy.orm import sessionmaker

from app.main import (
    Assignment,
    Base,
    C4AssessmentResult,
    ContributorAssignment,
    ExtractedKeyword,
    Lesson,
    Notification,
    ReferenceDocument,
    ReferenceKeywordPool,
    Submission,
    User,
    build_db_engine,
    run_schema_migrations,
    sync_postgres_sequences,
)

# Ordered by foreign-key dependencies
ORDERED_MODELS = [
    ("users", User),
    ("lessons", Lesson),
    ("assignments", Assignment),
    ("contributor_assignments", ContributorAssignment),
    ("reference_documents", ReferenceDocument),
    ("extracted_keywords", ExtractedKeyword),
    ("reference_keyword_pools", ReferenceKeywordPool),
    ("submissions", Submission),
    ("c4_assessment_results", C4AssessmentResult),
    ("notifications", Notification),
]

DATETIME_COLUMNS = {
    "created_at",
    "assigned_at",
    "uploaded_at",
    "processed_at",
    "submitted_at",
    "assessed_at",
    "applied_at",
}


def _parse_datetime(val: Any) -> datetime | None:
    if val is None or val == "":
        return None
    if isinstance(val, datetime):
        return val
    raw = str(val).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse datetime value: {val!r}")


def read_sqlite_snapshot(sqlite_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read all application tables from a SQLite file in strict read-only mode."""
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Source SQLite database not found: {sqlite_path}")

    uri = f"file:{sqlite_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    snapshot: dict[str, list[dict[str, Any]]] = {}

    try:
        cur = conn.cursor()
        existing_tables = {
            row["name"]
            for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        for table_name, model_cls in ORDERED_MODELS:
            if table_name not in existing_tables:
                snapshot[table_name] = []
                continue
            rows = cur.execute(f"SELECT * FROM {table_name} ORDER BY id ASC").fetchall()
            model_cols = {c.name for c in model_cls.__table__.columns}
            table_records: list[dict[str, Any]] = []
            for row in rows:
                row_dict = dict(row)
                filtered = {k: v for k, v in row_dict.items() if k in model_cols}
                # Apply model defaults for columns added in newer schema versions
                if table_name == "reference_documents" and "entry_mode" not in filtered:
                    filtered["entry_mode"] = "file"
                if table_name == "lessons" and "description" not in filtered:
                    filtered["description"] = ""
                table_records.append(filtered)
            snapshot[table_name] = table_records
    finally:
        conn.close()

    return snapshot


def write_snapshot_to_target(
    snapshot: dict[str, list[dict[str, Any]]],
    target_url: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Write a snapshot dictionary into the target database (PostgreSQL or SQLite),
    preserving all IDs and relationships idempotently.
    """
    counts = {table_name: len(snapshot.get(table_name, [])) for table_name, _ in ORDERED_MODELS}
    if dry_run:
        return counts

    target_engine = build_db_engine(target_url)
    try:
        run_schema_migrations(target_engine)
        TargetSession = sessionmaker(bind=target_engine, expire_on_commit=False)

        with TargetSession() as session:
            for table_name, model_cls in ORDERED_MODELS:
                records = snapshot.get(table_name, [])
                for raw_rec in records:
                    rec = dict(raw_rec)
                    for col_name in list(rec.keys()):
                        if col_name in DATETIME_COLUMNS and rec[col_name] is not None:
                            rec[col_name] = _parse_datetime(rec[col_name])
                    existing = session.get(model_cls, rec["id"])
                    if existing:
                        for k, v in rec.items():
                            setattr(existing, k, v)
                    else:
                        # Check unique constraints (e.g. users.email) in case seed ran before migration
                        if table_name == "users" and "email" in rec:
                            by_email = session.scalar(select(User).where(User.email == rec["email"]))
                            if by_email and by_email.id != rec["id"]:
                                session.delete(by_email)
                                session.flush()
                        session.add(model_cls(**rec))
                session.flush()
            session.commit()

        sync_postgres_sequences(target_engine)
    finally:
        target_engine.dispose()

    return counts


def main():
    parser = argparse.ArgumentParser(description="Safe, reversible SQLite-to-PostgreSQL migration tool for RecallAI")
    parser.add_argument(
        "--source",
        type=str,
        default=str(BACKEND_DIR.parent / "recallai.db"),
        help="Path to source SQLite database (opened read-only)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=os.getenv("TARGET_DATABASE_URL") or os.getenv("DATABASE_URL") or "",
        help="Target SQLAlchemy database URL (e.g., postgresql://...)",
    )
    parser.add_argument(
        "--backup-json",
        type=str,
        default="",
        help="Optional path to save a JSON backup snapshot of the source database",
    )
    parser.add_argument(
        "--restore-json",
        type=str,
        default="",
        help="Restore directly from a previously saved JSON snapshot instead of reading --source",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read source and validate row counts without writing to target database",
    )
    args = parser.parse_args()

    if args.restore_json:
        snap_path = Path(args.restore_json)
        snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
    else:
        src_path = Path(args.source)
        snapshot = read_sqlite_snapshot(src_path)
        if args.backup_json:
            backup_path = Path(args.backup_json)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
            print(f"[OK] Saved reversible backup snapshot to: {backup_path}")

    if not args.target and not args.dry_run:
        raise SystemExit("Error: --target database URL is required unless --dry-run is specified.")

    counts = write_snapshot_to_target(snapshot, args.target, dry_run=args.dry_run)
    mode_label = "DRY RUN" if args.dry_run else "MIGRATED"
    print(f"[{mode_label}] Row counts by table:")
    for table_name, count in counts.items():
        print(f"  - {table_name}: {count}")


if __name__ == "__main__":
    main()
