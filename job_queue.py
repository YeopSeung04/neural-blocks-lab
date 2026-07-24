import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timezone

from database import Database


class JobQueueConfigurationError(RuntimeError):
    pass


class JobExecutionError(RuntimeError):
    def __init__(self, message, code="job_failed"):
        super().__init__(message)
        self.code = code


def utc_time():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def job_id():
    return f"job_{secrets.token_hex(16)}"


class JobManager:
    queue_name = "neural-blocks:jobs"

    def __init__(
        self,
        database_target,
        *,
        mode=None,
        redis_url=None,
        payload_key=None,
        executor=None,
    ):
        self.database = Database(database_target)
        self.redis_url = redis_url or os.environ.get("NBL_REDIS_URL")
        self.mode = mode or os.environ.get(
            "NBL_JOB_MODE",
            "redis" if self.redis_url else "inline",
        )
        if self.mode not in ("inline", "redis"):
            raise JobQueueConfigurationError("NBL_JOB_MODE must be inline or redis")
        self.executor = executor
        self.redis = None
        if self.mode == "redis":
            if not self.redis_url:
                raise JobQueueConfigurationError(
                    "NBL_REDIS_URL is required when NBL_JOB_MODE=redis"
                )
            try:
                import redis
            except ImportError as error:
                raise JobQueueConfigurationError(
                    "Redis job mode requires the redis Python package"
                ) from error
            self.redis = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=10,
            )
        self.fernet = self._create_fernet(payload_key)
        self.ensure_schema()

    def _create_fernet(self, configured_key):
        key = configured_key or os.environ.get("NBL_JOB_PAYLOAD_KEY")
        if not key and self.mode == "inline":
            from cryptography.fernet import Fernet

            key = Fernet.generate_key().decode("ascii")
        if not key:
            return None
        try:
            from cryptography.fernet import Fernet

            return Fernet(key.encode("ascii"))
        except (ImportError, ValueError) as error:
            raise JobQueueConfigurationError(
                "NBL_JOB_PAYLOAD_KEY must be a valid Fernet key"
            ) from error

    def ensure_schema(self):
        with self.database.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS background_jobs (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT,
                    user_id TEXT,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dedupe_key TEXT,
                    payload_text TEXT NOT NULL,
                    payload_encrypted INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_background_jobs_tenant
                    ON background_jobs(tenant_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_background_jobs_status
                    ON background_jobs(status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_background_jobs_dedupe
                    ON background_jobs(dedupe_key, status);
                """
            )

    def set_executor(self, executor):
        self.executor = executor

    def health(self):
        if self.mode == "inline":
            return {"mode": "inline", "connected": True}
        try:
            connected = bool(self.redis.ping())
        except Exception:
            connected = False
        return {"mode": "redis", "connected": connected}

    def _encode_payload(self, payload, sensitive):
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if not sensitive:
            return raw, 0
        if not self.fernet:
            raise JobQueueConfigurationError(
                "NBL_JOB_PAYLOAD_KEY is required for sensitive queued jobs"
            )
        return self.fernet.encrypt(raw.encode("utf-8")).decode("ascii"), 1

    def _decode_payload(self, row):
        value = row["payload_text"]
        if row["payload_encrypted"]:
            if not self.fernet:
                raise JobQueueConfigurationError(
                    "Job payload key is unavailable in this worker"
                )
            value = self.fernet.decrypt(value.encode("ascii")).decode("utf-8")
        return json.loads(value)

    def enqueue(
        self,
        job_type,
        payload,
        *,
        tenant_id=None,
        user_id=None,
        dedupe_key=None,
        sensitive=False,
        max_attempts=3,
    ):
        now = utc_time()
        payload_text, encrypted = self._encode_payload(payload, sensitive)
        with self.database.connect() as db:
            if dedupe_key:
                existing = db.execute(
                    """
                    SELECT * FROM background_jobs
                    WHERE dedupe_key = ? AND status IN ('queued', 'running')
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (dedupe_key,),
                ).fetchone()
                if existing:
                    return self.public_job(existing)
            identifier = job_id()
            db.execute(
                """
                INSERT INTO background_jobs(
                    id, tenant_id, user_id, job_type, status, dedupe_key,
                    payload_text, payload_encrypted, result_json,
                    error_code, error_message, attempt, max_attempts,
                    created_at, updated_at, started_at, finished_at
                ) VALUES (
                    ?, ?, ?, ?, 'queued', ?, ?, ?, NULL,
                    NULL, NULL, 0, ?, ?, ?, NULL, NULL
                )
                """,
                (
                    identifier,
                    tenant_id,
                    user_id,
                    job_type,
                    dedupe_key,
                    payload_text,
                    encrypted,
                    max(1, min(10, int(max_attempts))),
                    now,
                    now,
                ),
            )
        if self.mode == "inline":
            self.process(identifier)
        else:
            self.redis.rpush(self.queue_name, identifier)
        return self.get(identifier)

    def get(self, identifier):
        with self.database.connect() as db:
            row = db.execute(
                "SELECT * FROM background_jobs WHERE id = ?",
                (identifier,),
            ).fetchone()
            return self.public_job(row) if row else None

    def get_for_auth(self, auth, identifier):
        job = self.get(identifier)
        if not job or job["tenantId"] != auth["user"]["tenantId"]:
            return None
        return job

    def list_for_auth(self, auth, limit=30):
        try:
            limit = max(1, min(100, int(limit)))
        except (TypeError, ValueError):
            limit = 30
        with self.database.connect() as db:
            rows = db.execute(
                """
                SELECT * FROM background_jobs
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (auth["user"]["tenantId"], limit),
            ).fetchall()
            return [self.public_job(row) for row in rows]

    def public_job(self, row):
        if not row:
            return None
        data = dict(row)
        result = json.loads(data["result_json"]) if data["result_json"] else None
        return {
            "id": data["id"],
            "tenantId": data["tenant_id"],
            "userId": data["user_id"],
            "jobType": data["job_type"],
            "status": data["status"],
            "attempt": data["attempt"],
            "maxAttempts": data["max_attempts"],
            "result": result,
            "error": (
                {
                    "code": data["error_code"] or "job_failed",
                    "message": data["error_message"] or "Background job failed",
                }
                if data["status"] == "failed"
                else None
            ),
            "createdAt": data["created_at"],
            "updatedAt": data["updated_at"],
            "startedAt": data["started_at"],
            "finishedAt": data["finished_at"],
        }

    def _claim(self, identifier):
        now = utc_time()
        with self.database.connect() as db:
            row = db.execute(
                "SELECT * FROM background_jobs WHERE id = ?",
                (identifier,),
            ).fetchone()
            if not row or row["status"] != "queued":
                return None
            updated = db.execute(
                """
                UPDATE background_jobs
                SET status = 'running',
                    attempt = attempt + 1,
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?,
                    error_code = NULL,
                    error_message = NULL
                WHERE id = ? AND status = 'queued'
                """,
                (now, now, identifier),
            )
            if getattr(updated, "rowcount", 0) != 1:
                return None
            claimed = db.execute(
                "SELECT * FROM background_jobs WHERE id = ?",
                (identifier,),
            ).fetchone()
            if not claimed or claimed["status"] != "running":
                return None
            return dict(claimed)

    def process(self, identifier):
        if not self.executor:
            raise JobQueueConfigurationError("Job executor is not configured")
        row = self._claim(identifier)
        if not row:
            return self.get(identifier)
        try:
            payload = self._decode_payload(row)
            result = self.executor(row["job_type"], payload)
            now = utc_time()
            with self.database.connect() as db:
                db.execute(
                    """
                    UPDATE background_jobs
                    SET status = 'succeeded',
                        result_json = ?,
                        payload_text = CASE
                            WHEN payload_encrypted = 1 THEN ''
                            ELSE payload_text
                        END,
                        updated_at = ?,
                        finished_at = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(
                            result or {},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        now,
                        now,
                        identifier,
                    ),
                )
        except Exception as error:
            code = getattr(error, "code", "job_failed")
            message = str(error)[:1000] or "Background job failed"
            retry = row["attempt"] < row["max_attempts"]
            now = utc_time()
            with self.database.connect() as db:
                db.execute(
                    """
                    UPDATE background_jobs
                    SET status = ?,
                        error_code = ?,
                        error_message = ?,
                        updated_at = ?,
                        finished_at = ?
                    WHERE id = ?
                    """,
                    (
                        "queued" if retry else "failed",
                        code,
                        message,
                        now,
                        None if retry else now,
                        identifier,
                    ),
                )
            if retry and self.mode == "redis":
                time.sleep(min(30, 2 ** row["attempt"]))
                self.redis.rpush(self.queue_name, identifier)
            elif retry:
                return self.process(identifier)
        return self.get(identifier)

    def work_once(self, timeout=5):
        if self.mode != "redis":
            raise JobQueueConfigurationError("Worker requires Redis job mode")
        item = self.redis.blpop(self.queue_name, timeout=timeout)
        if not item:
            return None
        return self.process(item[1])

    def recover_interrupted(self):
        now = utc_time()
        with self.database.connect() as db:
            rows = db.execute(
                """
                SELECT id FROM background_jobs
                WHERE status = 'running'
                """
            ).fetchall()
            db.execute(
                """
                UPDATE background_jobs
                SET status = 'queued',
                    error_code = 'worker_interrupted',
                    error_message = 'Worker stopped before completion',
                    updated_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
        if self.mode == "redis":
            for row in rows:
                self.redis.rpush(self.queue_name, row["id"])
        return len(rows)


class QueuedMailer:
    def __init__(self, job_manager):
        self.job_manager = job_manager

    @property
    def mode(self):
        return "queued" if self.job_manager.mode == "redis" else "inline"

    def send(self, recipient, subject, text, metadata=None):
        metadata = metadata or {}
        token = str(metadata.get("token") or "")
        dedupe_source = f"{recipient}:{metadata.get('kind')}:{token}"
        job = self.job_manager.enqueue(
            "email.send",
            {
                "recipient": recipient,
                "subject": subject,
                "text": text,
                "metadata": metadata,
            },
            dedupe_key="email:" + hashlib.sha256(
                dedupe_source.encode("utf-8")
            ).hexdigest(),
            sensitive=True,
            max_attempts=5,
        )
        return {"mode": self.mode, "jobId": job["id"]}
