import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


SESSION_DAYS = 7
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,38}[a-z0-9]$")
MODEL_FAMILIES = {"any", "mlp", "cnn", "rnn", "gan"}


class ApiError(Exception):
    def __init__(self, status, message, code="request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


def utc_now():
    return datetime.now(timezone.utc)


def iso_time(value=None):
    return (value or utc_now()).isoformat().replace("+00:00", "Z")


def parse_json(value, fallback=None):
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def required_text(value, label, maximum=200):
    text = str(value or "").strip()
    if not text:
        raise ApiError(400, f"{label} is required", "validation_error")
    if len(text) > maximum:
        raise ApiError(400, f"{label} is too long", "validation_error")
    return text


def normalize_email(value):
    email = str(value or "").strip().lower()
    if not EMAIL_PATTERN.match(email) or len(email) > 254:
        raise ApiError(400, "Valid email is required", "validation_error")
    return email


def password_digest(password, salt=None):
    password = str(password or "")
    if len(password) < 10:
        raise ApiError(400, "Password must contain at least 10 characters", "validation_error")
    if len(password) > 256:
        raise ApiError(400, "Password is too long", "validation_error")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        600_000,
        dklen=32,
    )
    return salt, digest


def session_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


def join_code(length=10):
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def row_dict(row):
    return dict(row) if row is not None else None


class NeuralBlocksBackend:
    def __init__(self, database_path):
        self.database_path = str(Path(database_path))
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self):
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self):
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    slug TEXT NOT NULL UNIQUE,
                    join_code TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'professor', 'student')),
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    csrf_token TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS courses (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                    owner_id TEXT NOT NULL REFERENCES users(id),
                    name TEXT NOT NULL,
                    code TEXT NOT NULL,
                    term TEXT NOT NULL,
                    join_code TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    UNIQUE(tenant_id, code, term)
                );

                CREATE TABLE IF NOT EXISTS enrollments (
                    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('instructor', 'student')),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(course_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS assignments (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                    created_by TEXT NOT NULL REFERENCES users(id),
                    title TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    due_at TEXT,
                    required_family TEXT NOT NULL,
                    target_accuracy REAL,
                    starter_snapshot_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                    owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS project_versions (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    snapshot_json TEXT NOT NULL,
                    saved_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS submissions (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                    assignment_id TEXT NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
                    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
                    student_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    attempt INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    auto_evaluation_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('submitted', 'graded')),
                    score REAL,
                    feedback TEXT NOT NULL DEFAULT '',
                    submitted_at TEXT NOT NULL,
                    graded_at TEXT,
                    graded_by TEXT REFERENCES users(id)
                );

                CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_courses_tenant ON courses(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_assignments_course ON assignments(tenant_id, course_id);
                CREATE INDEX IF NOT EXISTS idx_projects_course ON projects(tenant_id, course_id);
                CREATE INDEX IF NOT EXISTS idx_submissions_course ON submissions(tenant_id, course_id);
                """
            )

    def create_session(self, db, user_id):
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        created_at = utc_now()
        expires_at = created_at + timedelta(days=SESSION_DAYS)
        db.execute(
            """
            INSERT INTO sessions(token_hash, user_id, csrf_token, expires_at, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_hash(token), user_id, csrf, iso_time(expires_at), iso_time(created_at)),
        )
        return token, csrf

    def register(self, payload):
        email = normalize_email(payload.get("email"))
        display_name = required_text(payload.get("displayName"), "Display name", 80)
        password_salt, password_hash = password_digest(payload.get("password"))
        new_institution = bool(payload.get("createInstitution"))
        created_at = iso_time()
        with self.connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                if new_institution:
                    institution_name = required_text(
                        payload.get("institutionName"),
                        "Institution name",
                        120,
                    )
                    institution_slug = required_text(
                        payload.get("institutionSlug"),
                        "Institution slug",
                        40,
                    ).lower()
                    if not SLUG_PATTERN.match(institution_slug):
                        raise ApiError(
                            400,
                            "Institution slug must use lowercase letters, numbers, and hyphens",
                            "validation_error",
                        )
                    tenant = {
                        "id": new_id("tenant"),
                        "name": institution_name,
                        "slug": institution_slug,
                        "join_code": join_code(12),
                        "created_at": created_at,
                    }
                    db.execute(
                        """
                        INSERT INTO tenants(id, name, slug, join_code, created_at)
                        VALUES (:id, :name, :slug, :join_code, :created_at)
                        """,
                        tenant,
                    )
                    role = "admin"
                else:
                    tenant_join_code = required_text(
                        payload.get("institutionJoinCode"),
                        "Institution join code",
                        32,
                    ).upper()
                    tenant_row = db.execute(
                        "SELECT * FROM tenants WHERE join_code = ?",
                        (tenant_join_code,),
                    ).fetchone()
                    if not tenant_row:
                        raise ApiError(404, "Institution join code was not found", "not_found")
                    tenant = row_dict(tenant_row)
                    role = "student"

                user_id = new_id("user")
                db.execute(
                    """
                    INSERT INTO users(
                        id, tenant_id, email, display_name, role,
                        password_salt, password_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        tenant["id"],
                        email,
                        display_name,
                        role,
                        password_salt,
                        password_hash,
                        created_at,
                    ),
                )
                token, csrf = self.create_session(db, user_id)
                db.commit()
            except sqlite3.IntegrityError as error:
                db.rollback()
                if "users.email" in str(error):
                    raise ApiError(409, "Email is already registered", "conflict")
                if "tenants.slug" in str(error):
                    raise ApiError(409, "Institution slug is already used", "conflict")
                raise ApiError(409, "Registration data already exists", "conflict")
        auth = self.authenticate(token)
        auth["csrfToken"] = csrf
        auth["sessionToken"] = token
        return auth

    def login(self, payload):
        email = normalize_email(payload.get("email"))
        password = str(payload.get("password") or "")
        if len(password) < 10 or len(password) > 256:
            raise ApiError(401, "Email or password is incorrect", "invalid_credentials")
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,),
            ).fetchone()
            if not row:
                raise ApiError(401, "Email or password is incorrect", "invalid_credentials")
            _, digest = password_digest(password, row["password_salt"])
            if not hmac.compare_digest(digest, row["password_hash"]):
                raise ApiError(401, "Email or password is incorrect", "invalid_credentials")
            token, csrf = self.create_session(db, row["id"])
        auth = self.authenticate(token)
        auth["csrfToken"] = csrf
        auth["sessionToken"] = token
        return auth

    def authenticate(self, token):
        if not token:
            raise ApiError(401, "Authentication is required", "unauthorized")
        with self.connect() as db:
            row = db.execute(
                """
                SELECT
                    s.csrf_token, s.expires_at,
                    u.id AS user_id, u.email, u.display_name, u.role, u.tenant_id,
                    t.name AS tenant_name, t.slug AS tenant_slug, t.join_code AS tenant_join_code
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                JOIN tenants t ON t.id = u.tenant_id
                WHERE s.token_hash = ?
                """,
                (session_hash(token),),
            ).fetchone()
            if not row:
                raise ApiError(401, "Session is invalid", "unauthorized")
            expires_at = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
            if expires_at <= utc_now():
                db.execute(
                    "DELETE FROM sessions WHERE token_hash = ?",
                    (session_hash(token),),
                )
                raise ApiError(401, "Session has expired", "unauthorized")
            return {
                "user": {
                    "id": row["user_id"],
                    "email": row["email"],
                    "displayName": row["display_name"],
                    "role": row["role"],
                    "tenantId": row["tenant_id"],
                },
                "tenant": {
                    "id": row["tenant_id"],
                    "name": row["tenant_name"],
                    "slug": row["tenant_slug"],
                    "joinCode": row["tenant_join_code"]
                    if row["role"] in ("admin", "professor")
                    else None,
                },
                "csrfToken": row["csrf_token"],
            }

    def logout(self, token):
        if not token:
            return
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE token_hash = ?", (session_hash(token),))

    def require_role(self, auth, *roles):
        if auth["user"]["role"] not in roles:
            raise ApiError(403, "This action is not allowed for your role", "forbidden")

    def course_access(self, db, auth, course_id, instructor=False):
        row = db.execute(
            "SELECT * FROM courses WHERE id = ? AND tenant_id = ?",
            (course_id, auth["user"]["tenantId"]),
        ).fetchone()
        if not row:
            raise ApiError(404, "Course not found", "not_found")
        if auth["user"]["role"] == "admin":
            return row_dict(row)
        enrollment = db.execute(
            "SELECT role FROM enrollments WHERE course_id = ? AND user_id = ?",
            (course_id, auth["user"]["id"]),
        ).fetchone()
        if not enrollment or (instructor and enrollment["role"] != "instructor"):
            raise ApiError(403, "Course access is denied", "forbidden")
        return row_dict(row)

    def list_courses(self, auth):
        with self.connect() as db:
            if auth["user"]["role"] == "admin":
                rows = db.execute(
                    "SELECT * FROM courses WHERE tenant_id = ? ORDER BY created_at DESC",
                    (auth["user"]["tenantId"],),
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT c.*, e.role AS enrollment_role
                    FROM courses c
                    JOIN enrollments e ON e.course_id = c.id
                    WHERE c.tenant_id = ? AND e.user_id = ?
                    ORDER BY c.created_at DESC
                    """,
                    (auth["user"]["tenantId"], auth["user"]["id"]),
                ).fetchall()
            include_join_code = auth["user"]["role"] in ("admin", "professor")
            return [
                self.course_payload(row, include_join_code=include_join_code)
                for row in rows
            ]

    def create_course(self, auth, payload):
        self.require_role(auth, "admin", "professor")
        created_at = iso_time()
        course = {
            "id": new_id("course"),
            "tenant_id": auth["user"]["tenantId"],
            "owner_id": auth["user"]["id"],
            "name": required_text(payload.get("name"), "Course name", 120),
            "code": required_text(payload.get("code"), "Course code", 40).upper(),
            "term": required_text(payload.get("term"), "Term", 40),
            "join_code": join_code(10),
            "created_at": created_at,
        }
        with self.connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """
                    INSERT INTO courses(
                        id, tenant_id, owner_id, name, code, term, join_code, created_at
                    ) VALUES (
                        :id, :tenant_id, :owner_id, :name, :code, :term, :join_code, :created_at
                    )
                    """,
                    course,
                )
                db.execute(
                    """
                    INSERT INTO enrollments(course_id, user_id, role, created_at)
                    VALUES (?, ?, 'instructor', ?)
                    """,
                    (course["id"], auth["user"]["id"], created_at),
                )
                db.commit()
            except sqlite3.IntegrityError:
                db.rollback()
                raise ApiError(409, "Course code and term already exist", "conflict")
        return self.course_payload(course)

    def join_course(self, auth, payload):
        self.require_role(auth, "student")
        code = required_text(payload.get("joinCode"), "Course join code", 32).upper()
        with self.connect() as db:
            course = db.execute(
                "SELECT * FROM courses WHERE join_code = ? AND tenant_id = ?",
                (code, auth["user"]["tenantId"]),
            ).fetchone()
            if not course:
                raise ApiError(404, "Course join code was not found", "not_found")
            db.execute(
                """
                INSERT OR IGNORE INTO enrollments(course_id, user_id, role, created_at)
                VALUES (?, ?, 'student', ?)
                """,
                (course["id"], auth["user"]["id"], iso_time()),
            )
            return self.course_payload(course, include_join_code=False)

    def course_payload(self, row, include_join_code=True):
        data = row_dict(row)
        return {
            "id": data["id"],
            "name": data["name"],
            "code": data["code"],
            "term": data["term"],
            "joinCode": data["join_code"] if include_join_code else None,
            "ownerId": data["owner_id"],
            "enrollmentRole": data.get("enrollment_role"),
            "createdAt": data["created_at"],
        }

    def list_assignments(self, auth, course_id):
        with self.connect() as db:
            self.course_access(db, auth, course_id)
            rows = db.execute(
                """
                SELECT * FROM assignments
                WHERE tenant_id = ? AND course_id = ?
                ORDER BY created_at DESC
                """,
                (auth["user"]["tenantId"], course_id),
            ).fetchall()
            return [self.assignment_payload(row) for row in rows]

    def create_assignment(self, auth, course_id, payload):
        self.require_role(auth, "admin", "professor")
        with self.connect() as db:
            self.course_access(db, auth, course_id, instructor=True)
            required_family = str(payload.get("requiredFamily") or "any")
            if required_family not in MODEL_FAMILIES:
                raise ApiError(400, "Unsupported model family", "validation_error")
            target_accuracy = payload.get("targetAccuracy")
            if target_accuracy in ("", None):
                target_accuracy = None
            else:
                try:
                    target_accuracy = float(target_accuracy)
                except (TypeError, ValueError):
                    raise ApiError(
                        400,
                        "Target accuracy must be a number",
                        "validation_error",
                    )
                if target_accuracy < 0 or target_accuracy > 1:
                    raise ApiError(
                        400,
                        "Target accuracy must be between 0 and 1",
                        "validation_error",
                    )
            assignment = {
                "id": new_id("assignment"),
                "tenant_id": auth["user"]["tenantId"],
                "course_id": course_id,
                "created_by": auth["user"]["id"],
                "title": required_text(payload.get("title"), "Assignment title", 160),
                "instructions": str(payload.get("instructions") or "")[:5000],
                "due_at": payload.get("dueAt") or None,
                "required_family": required_family,
                "target_accuracy": target_accuracy,
                "starter_snapshot_json": compact_json(payload.get("starterSnapshot"))
                if payload.get("starterSnapshot")
                else None,
                "created_at": iso_time(),
            }
            db.execute(
                """
                INSERT INTO assignments(
                    id, tenant_id, course_id, created_by, title, instructions,
                    due_at, required_family, target_accuracy,
                    starter_snapshot_json, created_at
                ) VALUES (
                    :id, :tenant_id, :course_id, :created_by, :title, :instructions,
                    :due_at, :required_family, :target_accuracy,
                    :starter_snapshot_json, :created_at
                )
                """,
                assignment,
            )
            return self.assignment_payload(assignment)

    def assignment_payload(self, row):
        data = row_dict(row)
        return {
            "id": data["id"],
            "courseId": data["course_id"],
            "title": data["title"],
            "instructions": data["instructions"],
            "dueAt": data["due_at"],
            "requiredFamily": data["required_family"],
            "targetAccuracy": data["target_accuracy"],
            "starterSnapshot": parse_json(data["starter_snapshot_json"]),
            "createdBy": data["created_by"],
            "createdAt": data["created_at"],
        }

    def list_projects(self, auth, course_id):
        with self.connect() as db:
            course = self.course_access(db, auth, course_id)
            if auth["user"]["role"] in ("admin", "professor"):
                self.course_access(db, auth, course["id"], instructor=True)
                rows = db.execute(
                    """
                    SELECT p.*, u.display_name AS owner_name,
                        COUNT(v.id) AS version_count,
                        MAX(v.saved_at) AS latest_saved_at
                    FROM projects p
                    JOIN users u ON u.id = p.owner_id
                    LEFT JOIN project_versions v ON v.project_id = p.id
                    WHERE p.tenant_id = ? AND p.course_id = ?
                    GROUP BY p.id
                    ORDER BY p.updated_at DESC
                    """,
                    (auth["user"]["tenantId"], course_id),
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT p.*, u.display_name AS owner_name,
                        COUNT(v.id) AS version_count,
                        MAX(v.saved_at) AS latest_saved_at
                    FROM projects p
                    JOIN users u ON u.id = p.owner_id
                    LEFT JOIN project_versions v ON v.project_id = p.id
                    WHERE p.tenant_id = ? AND p.course_id = ? AND p.owner_id = ?
                    GROUP BY p.id
                    ORDER BY p.updated_at DESC
                    """,
                    (auth["user"]["tenantId"], course_id, auth["user"]["id"]),
                ).fetchall()
            return [self.project_payload(row) for row in rows]

    def save_project(self, auth, course_id, payload):
        with self.connect() as db:
            self.course_access(db, auth, course_id)
            project_id = payload.get("projectId")
            name = required_text(payload.get("name"), "Project name", 160)
            snapshot = payload.get("snapshot")
            if not isinstance(snapshot, dict):
                raise ApiError(400, "Experiment snapshot is required", "validation_error")
            now = iso_time()
            db.execute("BEGIN IMMEDIATE")
            if project_id:
                project = db.execute(
                    """
                    SELECT * FROM projects
                    WHERE id = ? AND tenant_id = ? AND course_id = ?
                    """,
                    (project_id, auth["user"]["tenantId"], course_id),
                ).fetchone()
                if not project:
                    raise ApiError(404, "Project not found", "not_found")
                if project["owner_id"] != auth["user"]["id"]:
                    raise ApiError(403, "Only the project owner can save it", "forbidden")
                db.execute(
                    "UPDATE projects SET name = ?, updated_at = ? WHERE id = ?",
                    (name, now, project_id),
                )
            else:
                project_id = new_id("project")
                db.execute(
                    """
                    INSERT INTO projects(
                        id, tenant_id, course_id, owner_id, name, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        auth["user"]["tenantId"],
                        course_id,
                        auth["user"]["id"],
                        name,
                        now,
                        now,
                    ),
                )
            version_id = new_id("version")
            db.execute(
                """
                INSERT INTO project_versions(id, tenant_id, project_id, snapshot_json, saved_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    auth["user"]["tenantId"],
                    project_id,
                    compact_json(snapshot),
                    now,
                ),
            )
            db.commit()
            row = db.execute(
                """
                SELECT p.*, u.display_name AS owner_name,
                    COUNT(v.id) AS version_count,
                    MAX(v.saved_at) AS latest_saved_at
                FROM projects p
                JOIN users u ON u.id = p.owner_id
                LEFT JOIN project_versions v ON v.project_id = p.id
                WHERE p.id = ? AND p.tenant_id = ?
                GROUP BY p.id
                """,
                (project_id, auth["user"]["tenantId"]),
            ).fetchone()
            result = self.project_payload(row)
            result["latestVersionId"] = version_id
            return result

    def get_project(self, auth, project_id):
        with self.connect() as db:
            project = db.execute(
                """
                SELECT p.*, u.display_name AS owner_name
                FROM projects p
                JOIN users u ON u.id = p.owner_id
                WHERE p.id = ? AND p.tenant_id = ?
                """,
                (project_id, auth["user"]["tenantId"]),
            ).fetchone()
            if not project:
                raise ApiError(404, "Project not found", "not_found")
            if project["owner_id"] != auth["user"]["id"]:
                self.course_access(db, auth, project["course_id"], instructor=True)
            version = db.execute(
                """
                SELECT * FROM project_versions
                WHERE project_id = ? AND tenant_id = ?
                ORDER BY saved_at DESC LIMIT 1
                """,
                (project_id, auth["user"]["tenantId"]),
            ).fetchone()
            result = self.project_payload(project)
            result["latestSnapshot"] = parse_json(version["snapshot_json"]) if version else None
            result["latestVersionId"] = version["id"] if version else None
            return result

    def project_payload(self, row):
        data = row_dict(row)
        return {
            "id": data["id"],
            "courseId": data["course_id"],
            "ownerId": data["owner_id"],
            "ownerName": data.get("owner_name"),
            "name": data["name"],
            "versionCount": int(data.get("version_count") or 0),
            "latestSavedAt": data.get("latest_saved_at"),
            "createdAt": data["created_at"],
            "updatedAt": data["updated_at"],
        }

    def evaluate_assignment(self, assignment, snapshot):
        checks = []
        family = (snapshot.get("model") or {}).get("family", "unknown")
        result = snapshot.get("result") or {}
        validation_accuracy = result.get("validationAccuracy")
        if assignment["required_family"] != "any":
            checks.append({
                "label": "Model family",
                "passed": family == assignment["required_family"],
                "detail": f"{family} / required {assignment['required_family']}",
            })
        if assignment["target_accuracy"] is not None:
            valid_accuracy = isinstance(validation_accuracy, (int, float))
            checks.append({
                "label": "Validation accuracy",
                "passed": valid_accuracy
                and validation_accuracy >= assignment["target_accuracy"],
                "detail": (
                    f"{validation_accuracy * 100:.1f}% / "
                    f"target {assignment['target_accuracy'] * 100:.1f}%"
                )
                if valid_accuracy
                else "No validation accuracy",
            })
        passed_count = sum(1 for check in checks if check["passed"])
        return {
            "passed": all(check["passed"] for check in checks),
            "checks": checks,
            "suggestedScore": round(passed_count / len(checks) * 100)
            if checks
            else None,
        }

    def submit_assignment(self, auth, assignment_id, payload):
        self.require_role(auth, "student")
        with self.connect() as db:
            assignment = db.execute(
                """
                SELECT * FROM assignments
                WHERE id = ? AND tenant_id = ?
                """,
                (assignment_id, auth["user"]["tenantId"]),
            ).fetchone()
            if not assignment:
                raise ApiError(404, "Assignment not found", "not_found")
            self.course_access(db, auth, assignment["course_id"])
            project_id = payload.get("projectId")
            if project_id:
                project = db.execute(
                    """
                    SELECT * FROM projects
                    WHERE id = ? AND tenant_id = ? AND course_id = ? AND owner_id = ?
                    """,
                    (
                        project_id,
                        auth["user"]["tenantId"],
                        assignment["course_id"],
                        auth["user"]["id"],
                    ),
                ).fetchone()
                if not project:
                    raise ApiError(404, "Project not found", "not_found")
            snapshot = payload.get("snapshot")
            if not isinstance(snapshot, dict):
                raise ApiError(400, "Experiment snapshot is required", "validation_error")
            attempt = db.execute(
                """
                SELECT COUNT(*) AS count FROM submissions
                WHERE tenant_id = ? AND assignment_id = ? AND student_id = ?
                """,
                (auth["user"]["tenantId"], assignment_id, auth["user"]["id"]),
            ).fetchone()["count"] + 1
            auto_evaluation = self.evaluate_assignment(assignment, snapshot)
            submission = {
                "id": new_id("submission"),
                "tenant_id": auth["user"]["tenantId"],
                "assignment_id": assignment_id,
                "course_id": assignment["course_id"],
                "project_id": project_id,
                "student_id": auth["user"]["id"],
                "attempt": attempt,
                "snapshot_json": compact_json(snapshot),
                "auto_evaluation_json": compact_json(auto_evaluation),
                "status": "submitted",
                "score": None,
                "feedback": "",
                "submitted_at": iso_time(),
                "graded_at": None,
                "graded_by": None,
            }
            db.execute(
                """
                INSERT INTO submissions(
                    id, tenant_id, assignment_id, course_id, project_id,
                    student_id, attempt, snapshot_json, auto_evaluation_json,
                    status, score, feedback, submitted_at, graded_at, graded_by
                ) VALUES (
                    :id, :tenant_id, :assignment_id, :course_id, :project_id,
                    :student_id, :attempt, :snapshot_json, :auto_evaluation_json,
                    :status, :score, :feedback, :submitted_at, :graded_at, :graded_by
                )
                """,
                submission,
            )
            row = db.execute(
                """
                SELECT s.*, u.display_name AS student_name
                FROM submissions s JOIN users u ON u.id = s.student_id
                WHERE s.id = ?
                """,
                (submission["id"],),
            ).fetchone()
            return self.submission_payload(row)

    def list_submissions(self, auth, course_id):
        with self.connect() as db:
            self.course_access(
                db,
                auth,
                course_id,
                instructor=auth["user"]["role"] in ("admin", "professor"),
            )
            parameters = [auth["user"]["tenantId"], course_id]
            student_filter = ""
            if auth["user"]["role"] == "student":
                student_filter = " AND s.student_id = ?"
                parameters.append(auth["user"]["id"])
            rows = db.execute(
                f"""
                SELECT s.*, u.display_name AS student_name
                FROM submissions s
                JOIN users u ON u.id = s.student_id
                WHERE s.tenant_id = ? AND s.course_id = ? {student_filter}
                ORDER BY s.submitted_at DESC
                """,
                parameters,
            ).fetchall()
            return [self.submission_payload(row) for row in rows]

    def grade_submission(self, auth, submission_id, payload):
        self.require_role(auth, "admin", "professor")
        try:
            score = float(payload.get("score"))
        except (TypeError, ValueError):
            raise ApiError(400, "Score must be a number", "validation_error")
        if score < 0 or score > 100:
            raise ApiError(400, "Score must be between 0 and 100", "validation_error")
        feedback = str(payload.get("feedback") or "")[:5000]
        with self.connect() as db:
            submission = db.execute(
                """
                SELECT * FROM submissions
                WHERE id = ? AND tenant_id = ?
                """,
                (submission_id, auth["user"]["tenantId"]),
            ).fetchone()
            if not submission:
                raise ApiError(404, "Submission not found", "not_found")
            self.course_access(db, auth, submission["course_id"], instructor=True)
            db.execute(
                """
                UPDATE submissions
                SET status = 'graded', score = ?, feedback = ?,
                    graded_at = ?, graded_by = ?
                WHERE id = ? AND tenant_id = ?
                """,
                (
                    round(score, 1),
                    feedback,
                    iso_time(),
                    auth["user"]["id"],
                    submission_id,
                    auth["user"]["tenantId"],
                ),
            )
            row = db.execute(
                """
                SELECT s.*, u.display_name AS student_name
                FROM submissions s JOIN users u ON u.id = s.student_id
                WHERE s.id = ?
                """,
                (submission_id,),
            ).fetchone()
            return self.submission_payload(row)

    def submission_payload(self, row):
        data = row_dict(row)
        return {
            "id": data["id"],
            "assignmentId": data["assignment_id"],
            "courseId": data["course_id"],
            "projectId": data["project_id"],
            "studentId": data["student_id"],
            "studentName": data.get("student_name"),
            "attempt": data["attempt"],
            "snapshot": parse_json(data["snapshot_json"], {}),
            "autoEvaluation": parse_json(data["auto_evaluation_json"], {}),
            "status": data["status"],
            "score": data["score"],
            "feedback": data["feedback"],
            "submittedAt": data["submitted_at"],
            "gradedAt": data["graded_at"],
            "gradedBy": data["graded_by"],
        }
