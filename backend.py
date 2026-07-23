import hashlib
import hmac
import json
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from database import Database, DatabaseIntegrityError
from mailer import Mailer


SESSION_DAYS = 7
EMAIL_VERIFICATION_HOURS = 24
PASSWORD_RESET_MINUTES = 60
INVITATION_DAYS = 7
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
    def __init__(
        self,
        database_target,
        mailer=None,
        base_url="http://127.0.0.1:8770",
        expose_dev_tokens=True,
    ):
        self.database = Database(database_target)
        self.database_path = self.database.target
        self.mailer = mailer or Mailer()
        self.base_url = base_url.rstrip("/")
        self.expose_dev_tokens = expose_dev_tokens
        self.initialize()

    def connect(self):
        return self.database.connect()

    def initialize(self):
        previous_user_columns = self.database.column_names("users")
        blob_type = "BYTEA" if self.database.engine == "postgres" else "BLOB"
        with self.connect() as db:
            db.executescript(
                f"""
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
                    password_salt {blob_type} NOT NULL,
                    password_hash {blob_type} NOT NULL,
                    email_verified_at TEXT,
                    auth_provider TEXT NOT NULL DEFAULT 'password',
                    external_subject TEXT,
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

                CREATE TABLE IF NOT EXISTS invitations (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                    email TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('professor', 'student')),
                    course_id TEXT REFERENCES courses(id) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_by TEXT NOT NULL REFERENCES users(id),
                    expires_at TEXT NOT NULL,
                    accepted_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS email_tokens (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    purpose TEXT NOT NULL CHECK(purpose IN ('verify_email', 'password_reset')),
                    token_hash TEXT NOT NULL UNIQUE,
                    expires_at TEXT NOT NULL,
                    used_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT REFERENCES tenants(id) ON DELETE CASCADE,
                    user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
                    event_type TEXT NOT NULL,
                    entity_type TEXT,
                    entity_id TEXT,
                    metadata_json TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS identity_providers (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL CHECK(kind IN ('oidc', 'lti')),
                    name TEXT NOT NULL,
                    issuer TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    authorization_endpoint TEXT NOT NULL,
                    token_endpoint TEXT,
                    jwks_uri TEXT NOT NULL,
                    client_secret_env TEXT,
                    service_token_auth_method TEXT NOT NULL DEFAULT 'client_secret_basic',
                    private_key_env TEXT,
                    private_key_kid TEXT,
                    deployment_id TEXT,
                    default_role TEXT NOT NULL CHECK(default_role IN ('professor', 'student')),
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_by TEXT NOT NULL REFERENCES users(id),
                    created_at TEXT NOT NULL,
                    UNIQUE(tenant_id, issuer, client_id)
                );

                CREATE TABLE IF NOT EXISTS federation_states (
                    state_hash TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                    provider_id TEXT NOT NULL REFERENCES identity_providers(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL CHECK(kind IN ('oidc', 'lti')),
                    nonce TEXT NOT NULL,
                    target_path TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS external_identities (
                    provider_id TEXT NOT NULL REFERENCES identity_providers(id) ON DELETE CASCADE,
                    subject TEXT NOT NULL,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(provider_id, subject)
                );

                CREATE TABLE IF NOT EXISTS lti_contexts (
                    provider_id TEXT NOT NULL REFERENCES identity_providers(id) ON DELETE CASCADE,
                    context_id TEXT NOT NULL,
                    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                    resource_link_id TEXT,
                    nrps_memberships_url TEXT,
                    nrps_scope_json TEXT NOT NULL DEFAULT '[]',
                    ags_lineitems_url TEXT,
                    ags_lineitem_url TEXT,
                    ags_scope_json TEXT NOT NULL DEFAULT '[]',
                    last_roster_sync_at TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(provider_id, context_id)
                );

                CREATE TABLE IF NOT EXISTS lti_memberships (
                    provider_id TEXT NOT NULL REFERENCES identity_providers(id) ON DELETE CASCADE,
                    context_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    roles_json TEXT NOT NULL,
                    synced_at TEXT NOT NULL,
                    PRIMARY KEY(provider_id, context_id, subject)
                );

                CREATE TABLE IF NOT EXISTS lti_line_items (
                    assignment_id TEXT PRIMARY KEY REFERENCES assignments(id) ON DELETE CASCADE,
                    provider_id TEXT NOT NULL REFERENCES identity_providers(id) ON DELETE CASCADE,
                    context_id TEXT NOT NULL,
                    lineitem_url TEXT NOT NULL,
                    score_maximum REAL NOT NULL DEFAULT 100,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lti_grade_passbacks (
                    id TEXT PRIMARY KEY,
                    submission_id TEXT NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
                    provider_id TEXT NOT NULL REFERENCES identity_providers(id) ON DELETE CASCADE,
                    lineitem_url TEXT NOT NULL,
                    score_given REAL NOT NULL,
                    response_json TEXT NOT NULL,
                    sent_by TEXT NOT NULL REFERENCES users(id),
                    sent_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_courses_tenant ON courses(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_assignments_course ON assignments(tenant_id, course_id);
                CREATE INDEX IF NOT EXISTS idx_projects_course ON projects(tenant_id, course_id);
                CREATE INDEX IF NOT EXISTS idx_submissions_course ON submissions(tenant_id, course_id);
                CREATE INDEX IF NOT EXISTS idx_invitations_tenant ON invitations(tenant_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_email_tokens_user ON email_tokens(user_id, purpose);
                CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_events(tenant_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_identity_providers_tenant ON identity_providers(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_lti_memberships_user ON lti_memberships(user_id);
                CREATE INDEX IF NOT EXISTS idx_lti_passbacks_submission ON lti_grade_passbacks(submission_id, sent_at);
                """
            )
        self.database.add_column_if_missing("users", "email_verified_at", "TEXT")
        self.database.add_column_if_missing(
            "users",
            "auth_provider",
            "TEXT NOT NULL DEFAULT 'password'",
        )
        self.database.add_column_if_missing("users", "external_subject", "TEXT")
        self.database.add_column_if_missing(
            "identity_providers",
            "service_token_auth_method",
            "TEXT NOT NULL DEFAULT 'client_secret_basic'",
        )
        self.database.add_column_if_missing(
            "identity_providers",
            "private_key_env",
            "TEXT",
        )
        self.database.add_column_if_missing(
            "identity_providers",
            "private_key_kid",
            "TEXT",
        )
        self.database.add_column_if_missing(
            "lti_contexts",
            "resource_link_id",
            "TEXT",
        )
        self.database.add_column_if_missing(
            "lti_contexts",
            "nrps_memberships_url",
            "TEXT",
        )
        self.database.add_column_if_missing(
            "lti_contexts",
            "nrps_scope_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )
        self.database.add_column_if_missing(
            "lti_contexts",
            "ags_lineitems_url",
            "TEXT",
        )
        self.database.add_column_if_missing(
            "lti_contexts",
            "ags_lineitem_url",
            "TEXT",
        )
        self.database.add_column_if_missing(
            "lti_contexts",
            "ags_scope_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )
        self.database.add_column_if_missing(
            "lti_contexts",
            "last_roster_sync_at",
            "TEXT",
        )
        if previous_user_columns and "email_verified_at" not in previous_user_columns:
            with self.connect() as db:
                db.execute(
                    """
                    UPDATE users
                    SET email_verified_at = created_at
                    WHERE email_verified_at IS NULL AND created_at IS NOT NULL
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

    def record_audit(
        self,
        db,
        event_type,
        tenant_id=None,
        user_id=None,
        entity_type=None,
        entity_id=None,
        metadata=None,
        context=None,
    ):
        context = context or {}
        db.execute(
            """
            INSERT INTO audit_events(
                id, tenant_id, user_id, event_type, entity_type, entity_id,
                metadata_json, ip_address, user_agent, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id("audit"),
                tenant_id,
                user_id,
                event_type,
                entity_type,
                entity_id,
                compact_json(metadata or {}),
                str(context.get("ip") or "")[:80] or None,
                str(context.get("userAgent") or "")[:500] or None,
                iso_time(),
            ),
        )

    def create_email_token(self, db, user_id, purpose, lifetime):
        token = secrets.token_urlsafe(32)
        now = utc_now()
        db.execute(
            """
            UPDATE email_tokens
            SET used_at = ?
            WHERE user_id = ? AND purpose = ? AND used_at IS NULL
            """,
            (iso_time(now), user_id, purpose),
        )
        db.execute(
            """
            INSERT INTO email_tokens(
                id, user_id, purpose, token_hash, expires_at, used_at, created_at
            ) VALUES (?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                new_id("email_token"),
                user_id,
                purpose,
                session_hash(token),
                iso_time(now + lifetime),
                iso_time(now),
            ),
        )
        return token

    def send_verification_email(self, email, display_name, token):
        link = f"{self.base_url}/?verify_email={token}"
        return self.mailer.send(
            email,
            "Neural Blocks Lab 이메일 인증",
            (
                f"{display_name}님, 아래 링크에서 이메일 인증을 완료하세요.\n\n"
                f"{link}\n\n"
                f"인증 토큰: {token}\n"
                f"이 링크는 {EMAIL_VERIFICATION_HOURS}시간 동안 유효합니다."
            ),
            {"kind": "verify_email", "token": token, "link": link},
        )

    def send_password_reset_email(self, email, display_name, token):
        link = f"{self.base_url}/?password_reset={token}"
        return self.mailer.send(
            email,
            "Neural Blocks Lab 비밀번호 재설정",
            (
                f"{display_name}님, 아래 링크에서 비밀번호를 재설정하세요.\n\n"
                f"{link}\n\n"
                f"재설정 토큰: {token}\n"
                f"이 링크는 {PASSWORD_RESET_MINUTES}분 동안 유효합니다."
            ),
            {"kind": "password_reset", "token": token, "link": link},
        )

    def register(self, payload, request_context=None):
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
                        password_salt, password_hash, email_verified_at,
                        auth_provider, external_subject, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'password', NULL, ?)
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
                verification_token = self.create_email_token(
                    db,
                    user_id,
                    "verify_email",
                    timedelta(hours=EMAIL_VERIFICATION_HOURS),
                )
                token, csrf = self.create_session(db, user_id)
                self.record_audit(
                    db,
                    "user.registered",
                    tenant["id"],
                    user_id,
                    "user",
                    user_id,
                    {"role": role, "email": email},
                    request_context,
                )
                db.commit()
            except DatabaseIntegrityError as error:
                db.rollback()
                error_text = str(error).lower()
                if "users.email" in error_text or "email" in error_text:
                    raise ApiError(409, "Email is already registered", "conflict")
                if "tenants.slug" in error_text or "slug" in error_text:
                    raise ApiError(409, "Institution slug is already used", "conflict")
                raise ApiError(409, "Registration data already exists", "conflict")
        delivery = self.send_verification_email(email, display_name, verification_token)
        auth = self.authenticate(token)
        auth["csrfToken"] = csrf
        auth["sessionToken"] = token
        auth["verificationRequired"] = True
        auth["mailDelivery"] = delivery["mode"]
        if self.expose_dev_tokens:
            auth["devVerificationToken"] = verification_token
        return auth

    def login(self, payload, request_context=None):
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
            self.record_audit(
                db,
                "user.login",
                row["tenant_id"],
                row["id"],
                "session",
                None,
                {"provider": "password"},
                request_context,
            )
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
                    u.email_verified_at, u.auth_provider,
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
                    "emailVerified": bool(row["email_verified_at"]),
                    "authProvider": row["auth_provider"],
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

    def require_verified(self, auth):
        if not auth["user"].get("emailVerified"):
            raise ApiError(
                403,
                "Email verification is required before changing classroom data",
                "email_not_verified",
            )

    def verify_email(self, payload, request_context=None):
        token = required_text(payload.get("token"), "Verification token", 200)
        now = utc_now()
        with self.connect() as db:
            row = db.execute(
                """
                SELECT et.*, u.tenant_id, u.email
                FROM email_tokens et
                JOIN users u ON u.id = et.user_id
                WHERE et.token_hash = ? AND et.purpose = 'verify_email'
                """,
                (session_hash(token),),
            ).fetchone()
            if not row or row["used_at"]:
                raise ApiError(400, "Verification token is invalid", "invalid_token")
            expires_at = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
            if expires_at <= now:
                raise ApiError(400, "Verification token has expired", "expired_token")
            db.execute(
                "UPDATE users SET email_verified_at = ? WHERE id = ?",
                (iso_time(now), row["user_id"]),
            )
            db.execute(
                "UPDATE email_tokens SET used_at = ? WHERE id = ?",
                (iso_time(now), row["id"]),
            )
            self.record_audit(
                db,
                "user.email_verified",
                row["tenant_id"],
                row["user_id"],
                "user",
                row["user_id"],
                {"email": row["email"]},
                request_context,
            )
        return {"status": "verified"}

    def resend_verification(self, payload):
        email = str(payload.get("email") or "").strip().lower()
        response = {"status": "accepted"}
        if not EMAIL_PATTERN.match(email):
            return response
        with self.connect() as db:
            user = db.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,),
            ).fetchone()
            if not user or user["email_verified_at"]:
                return response
            token = self.create_email_token(
                db,
                user["id"],
                "verify_email",
                timedelta(hours=EMAIL_VERIFICATION_HOURS),
            )
        delivery = self.send_verification_email(email, user["display_name"], token)
        response["mailDelivery"] = delivery["mode"]
        if self.expose_dev_tokens:
            response["devVerificationToken"] = token
        return response

    def request_password_reset(self, payload):
        email = str(payload.get("email") or "").strip().lower()
        response = {"status": "accepted"}
        if not EMAIL_PATTERN.match(email):
            return response
        with self.connect() as db:
            user = db.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,),
            ).fetchone()
            if not user:
                return response
            token = self.create_email_token(
                db,
                user["id"],
                "password_reset",
                timedelta(minutes=PASSWORD_RESET_MINUTES),
            )
        delivery = self.send_password_reset_email(email, user["display_name"], token)
        response["mailDelivery"] = delivery["mode"]
        if self.expose_dev_tokens:
            response["devPasswordResetToken"] = token
        return response

    def confirm_password_reset(self, payload, request_context=None):
        token = required_text(payload.get("token"), "Password reset token", 200)
        password_salt, password_hash = password_digest(payload.get("password"))
        now = utc_now()
        with self.connect() as db:
            row = db.execute(
                """
                SELECT et.*, u.tenant_id
                FROM email_tokens et
                JOIN users u ON u.id = et.user_id
                WHERE et.token_hash = ? AND et.purpose = 'password_reset'
                """,
                (session_hash(token),),
            ).fetchone()
            if not row or row["used_at"]:
                raise ApiError(400, "Password reset token is invalid", "invalid_token")
            expires_at = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
            if expires_at <= now:
                raise ApiError(400, "Password reset token has expired", "expired_token")
            db.execute(
                """
                UPDATE users
                SET password_salt = ?, password_hash = ?, email_verified_at = ?
                WHERE id = ?
                """,
                (password_salt, password_hash, iso_time(now), row["user_id"]),
            )
            db.execute(
                "UPDATE email_tokens SET used_at = ? WHERE id = ?",
                (iso_time(now), row["id"]),
            )
            db.execute("DELETE FROM sessions WHERE user_id = ?", (row["user_id"],))
            self.record_audit(
                db,
                "user.password_reset",
                row["tenant_id"],
                row["user_id"],
                "user",
                row["user_id"],
                {},
                request_context,
            )
        return {"status": "password_reset"}

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
        self.require_verified(auth)
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
                self.record_audit(
                    db,
                    "course.created",
                    auth["user"]["tenantId"],
                    auth["user"]["id"],
                    "course",
                    course["id"],
                    {"code": course["code"], "term": course["term"]},
                    auth.get("_request"),
                )
                db.commit()
            except DatabaseIntegrityError:
                db.rollback()
                raise ApiError(409, "Course code and term already exist", "conflict")
        return self.course_payload(course)

    def join_course(self, auth, payload):
        self.require_role(auth, "student")
        self.require_verified(auth)
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
                INSERT INTO enrollments(course_id, user_id, role, created_at)
                VALUES (?, ?, 'student', ?)
                ON CONFLICT(course_id, user_id) DO NOTHING
                """,
                (course["id"], auth["user"]["id"], iso_time()),
            )
            self.record_audit(
                db,
                "course.joined",
                auth["user"]["tenantId"],
                auth["user"]["id"],
                "course",
                course["id"],
                {"role": "student"},
                auth.get("_request"),
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

    def send_invitation_email(self, email, role, tenant_name, token):
        link = f"{self.base_url}/?invitation={token}"
        role_label = "교수" if role == "professor" else "학생"
        return self.mailer.send(
            email,
            f"{tenant_name} Neural Blocks Lab 초대",
            (
                f"{tenant_name}에서 {role_label} 계정으로 초대했습니다.\n\n"
                f"{link}\n\n"
                f"초대 토큰: {token}\n"
                f"이 초대는 {INVITATION_DAYS}일 동안 유효합니다."
            ),
            {"kind": "invitation", "token": token, "link": link, "role": role},
        )

    def create_invitation(self, auth, payload):
        self.require_role(auth, "admin")
        self.require_verified(auth)
        email = normalize_email(payload.get("email"))
        role = str(payload.get("role") or "professor")
        if role not in ("professor", "student"):
            raise ApiError(400, "Invitation role is invalid", "validation_error")
        course_id = str(payload.get("courseId") or "").strip() or None
        token = secrets.token_urlsafe(32)
        now = utc_now()
        invitation = {
            "id": new_id("invitation"),
            "tenant_id": auth["user"]["tenantId"],
            "email": email,
            "role": role,
            "course_id": course_id,
            "token_hash": session_hash(token),
            "created_by": auth["user"]["id"],
            "expires_at": iso_time(now + timedelta(days=INVITATION_DAYS)),
            "created_at": iso_time(now),
        }
        with self.connect() as db:
            if course_id:
                self.course_access(db, auth, course_id, instructor=True)
            existing_user = db.execute(
                "SELECT id FROM users WHERE email = ?",
                (email,),
            ).fetchone()
            if existing_user:
                raise ApiError(409, "Email is already registered", "conflict")
            db.execute(
                """
                UPDATE invitations
                SET accepted_at = ?
                WHERE tenant_id = ? AND email = ? AND accepted_at IS NULL
                """,
                (iso_time(now), auth["user"]["tenantId"], email),
            )
            db.execute(
                """
                INSERT INTO invitations(
                    id, tenant_id, email, role, course_id, token_hash,
                    created_by, expires_at, accepted_at, created_at
                ) VALUES (
                    :id, :tenant_id, :email, :role, :course_id, :token_hash,
                    :created_by, :expires_at, NULL, :created_at
                )
                """,
                invitation,
            )
            self.record_audit(
                db,
                "invitation.created",
                auth["user"]["tenantId"],
                auth["user"]["id"],
                "invitation",
                invitation["id"],
                {"email": email, "role": role, "courseId": course_id},
                auth.get("_request"),
            )
        delivery = self.send_invitation_email(
            email,
            role,
            auth["tenant"]["name"],
            token,
        )
        result = {
            "id": invitation["id"],
            "email": email,
            "role": role,
            "courseId": course_id,
            "expiresAt": invitation["expires_at"],
            "acceptedAt": None,
            "mailDelivery": delivery["mode"],
        }
        if self.expose_dev_tokens:
            result["devInvitationToken"] = token
        return result

    def list_invitations(self, auth):
        self.require_role(auth, "admin")
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT i.*, c.name AS course_name, u.display_name AS created_by_name
                FROM invitations i
                LEFT JOIN courses c ON c.id = i.course_id
                JOIN users u ON u.id = i.created_by
                WHERE i.tenant_id = ?
                ORDER BY i.created_at DESC
                LIMIT 100
                """,
                (auth["user"]["tenantId"],),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "email": row["email"],
                    "role": row["role"],
                    "courseId": row["course_id"],
                    "courseName": row["course_name"],
                    "createdByName": row["created_by_name"],
                    "expiresAt": row["expires_at"],
                    "acceptedAt": row["accepted_at"],
                    "createdAt": row["created_at"],
                }
                for row in rows
            ]

    def accept_invitation(self, payload, request_context=None):
        token = required_text(payload.get("token"), "Invitation token", 200)
        display_name = required_text(payload.get("displayName"), "Display name", 80)
        password_salt, password_hash = password_digest(payload.get("password"))
        now = utc_now()
        with self.connect() as db:
            invitation = db.execute(
                """
                SELECT i.*, t.name AS tenant_name, t.slug AS tenant_slug
                FROM invitations i
                JOIN tenants t ON t.id = i.tenant_id
                WHERE i.token_hash = ?
                """,
                (session_hash(token),),
            ).fetchone()
            if not invitation or invitation["accepted_at"]:
                raise ApiError(400, "Invitation token is invalid", "invalid_token")
            expires_at = datetime.fromisoformat(
                invitation["expires_at"].replace("Z", "+00:00")
            )
            if expires_at <= now:
                raise ApiError(400, "Invitation token has expired", "expired_token")
            user_id = new_id("user")
            try:
                db.execute(
                    """
                    INSERT INTO users(
                        id, tenant_id, email, display_name, role,
                        password_salt, password_hash, email_verified_at,
                        auth_provider, external_subject, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'password', NULL, ?)
                    """,
                    (
                        user_id,
                        invitation["tenant_id"],
                        invitation["email"],
                        display_name,
                        invitation["role"],
                        password_salt,
                        password_hash,
                        iso_time(now),
                        iso_time(now),
                    ),
                )
            except DatabaseIntegrityError:
                raise ApiError(409, "Email is already registered", "conflict")
            if invitation["course_id"]:
                enrollment_role = (
                    "instructor" if invitation["role"] == "professor" else "student"
                )
                db.execute(
                    """
                    INSERT INTO enrollments(course_id, user_id, role, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(course_id, user_id) DO NOTHING
                    """,
                    (
                        invitation["course_id"],
                        user_id,
                        enrollment_role,
                        iso_time(now),
                    ),
                )
            db.execute(
                "UPDATE invitations SET accepted_at = ? WHERE id = ?",
                (iso_time(now), invitation["id"]),
            )
            session_token, csrf = self.create_session(db, user_id)
            self.record_audit(
                db,
                "invitation.accepted",
                invitation["tenant_id"],
                user_id,
                "invitation",
                invitation["id"],
                {"role": invitation["role"], "courseId": invitation["course_id"]},
                request_context,
            )
        auth = self.authenticate(session_token)
        auth["csrfToken"] = csrf
        auth["sessionToken"] = session_token
        return auth

    def list_course_members(self, auth, course_id):
        self.require_role(auth, "admin", "professor")
        with self.connect() as db:
            self.course_access(db, auth, course_id, instructor=True)
            rows = db.execute(
                """
                SELECT
                    u.id, u.email, u.display_name, u.role AS tenant_role,
                    u.email_verified_at, u.auth_provider,
                    e.role AS course_role, e.created_at AS enrolled_at,
                    COUNT(s.id) AS submission_count,
                    MAX(s.submitted_at) AS last_submission_at
                FROM enrollments e
                JOIN users u ON u.id = e.user_id
                LEFT JOIN submissions s
                    ON s.course_id = e.course_id AND s.student_id = u.id
                WHERE e.course_id = ? AND u.tenant_id = ?
                GROUP BY
                    u.id, u.email, u.display_name, u.role,
                    u.email_verified_at, u.auth_provider,
                    e.role, e.created_at
                ORDER BY e.role, u.display_name
                """,
                (course_id, auth["user"]["tenantId"]),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "email": row["email"],
                    "displayName": row["display_name"],
                    "tenantRole": row["tenant_role"],
                    "courseRole": row["course_role"],
                    "emailVerified": bool(row["email_verified_at"]),
                    "authProvider": row["auth_provider"],
                    "enrolledAt": row["enrolled_at"],
                    "submissionCount": int(row["submission_count"] or 0),
                    "lastSubmissionAt": row["last_submission_at"],
                }
                for row in rows
            ]

    def remove_course_member(self, auth, course_id, user_id):
        self.require_role(auth, "admin", "professor")
        self.require_verified(auth)
        with self.connect() as db:
            self.course_access(db, auth, course_id, instructor=True)
            enrollment = db.execute(
                """
                SELECT e.role, u.display_name
                FROM enrollments e
                JOIN users u ON u.id = e.user_id
                WHERE e.course_id = ? AND e.user_id = ? AND u.tenant_id = ?
                """,
                (course_id, user_id, auth["user"]["tenantId"]),
            ).fetchone()
            if not enrollment:
                raise ApiError(404, "Course member was not found", "not_found")
            if enrollment["role"] != "student":
                raise ApiError(400, "Instructor enrollment cannot be removed here", "validation_error")
            db.execute(
                "DELETE FROM enrollments WHERE course_id = ? AND user_id = ?",
                (course_id, user_id),
            )
            self.record_audit(
                db,
                "course.member_removed",
                auth["user"]["tenantId"],
                auth["user"]["id"],
                "course",
                course_id,
                {"removedUserId": user_id, "displayName": enrollment["display_name"]},
                auth.get("_request"),
            )
        return {"status": "removed"}

    def list_audit_events(self, auth, limit=100):
        self.require_role(auth, "admin")
        try:
            limit = max(1, min(200, int(limit)))
        except (TypeError, ValueError):
            limit = 100
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT a.*, u.display_name AS user_name
                FROM audit_events a
                LEFT JOIN users u ON u.id = a.user_id
                WHERE a.tenant_id = ?
                ORDER BY a.created_at DESC
                LIMIT ?
                """,
                (auth["user"]["tenantId"], limit),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "eventType": row["event_type"],
                    "userId": row["user_id"],
                    "userName": row["user_name"],
                    "entityType": row["entity_type"],
                    "entityId": row["entity_id"],
                    "metadata": parse_json(row["metadata_json"], {}),
                    "ipAddress": row["ip_address"],
                    "createdAt": row["created_at"],
                }
                for row in rows
            ]

    def create_identity_provider(self, auth, payload):
        self.require_role(auth, "admin")
        self.require_verified(auth)
        kind = str(payload.get("kind") or "oidc").lower()
        if kind not in ("oidc", "lti"):
            raise ApiError(400, "Provider kind must be oidc or lti", "validation_error")
        default_role = str(payload.get("defaultRole") or "student").lower()
        if default_role not in ("professor", "student"):
            raise ApiError(400, "Default role is invalid", "validation_error")
        client_secret_env = str(payload.get("clientSecretEnv") or "").strip() or None
        if client_secret_env and not re.fullmatch(r"[A-Z][A-Z0-9_]{2,80}", client_secret_env):
            raise ApiError(
                400,
                "Client secret environment variable name is invalid",
                "validation_error",
            )
        service_token_auth_method = str(
            payload.get("serviceTokenAuthMethod") or "client_secret_basic"
        ).strip()
        if service_token_auth_method not in ("client_secret_basic", "private_key_jwt"):
            raise ApiError(
                400,
                "LTI service token authentication method is invalid",
                "validation_error",
            )
        private_key_env = str(payload.get("privateKeyEnv") or "").strip() or None
        if private_key_env and not re.fullmatch(r"[A-Z][A-Z0-9_]{2,80}", private_key_env):
            raise ApiError(
                400,
                "Private key environment variable name is invalid",
                "validation_error",
            )
        provider = {
            "id": new_id("provider"),
            "tenant_id": auth["user"]["tenantId"],
            "kind": kind,
            "name": required_text(payload.get("name"), "Provider name", 100),
            "issuer": required_text(payload.get("issuer"), "Issuer", 500).rstrip("/"),
            "client_id": required_text(payload.get("clientId"), "Client ID", 300),
            "authorization_endpoint": required_text(
                payload.get("authorizationEndpoint"),
                "Authorization endpoint",
                1000,
            ),
            "token_endpoint": str(payload.get("tokenEndpoint") or "").strip() or None,
            "jwks_uri": required_text(payload.get("jwksUri"), "JWKS URI", 1000),
            "client_secret_env": client_secret_env,
            "service_token_auth_method": service_token_auth_method,
            "private_key_env": private_key_env,
            "private_key_kid": str(payload.get("privateKeyKid") or "").strip() or None,
            "deployment_id": str(payload.get("deploymentId") or "").strip() or None,
            "default_role": default_role,
            "enabled": 1 if payload.get("enabled", True) else 0,
            "created_by": auth["user"]["id"],
            "created_at": iso_time(),
        }
        if kind == "oidc" and not provider["token_endpoint"]:
            raise ApiError(400, "OIDC token endpoint is required", "validation_error")
        if kind == "lti" and not provider["deployment_id"]:
            raise ApiError(400, "LTI deployment ID is required", "validation_error")
        if (
            kind == "lti"
            and service_token_auth_method == "private_key_jwt"
            and not private_key_env
        ):
            raise ApiError(
                400,
                "Private key environment variable is required",
                "validation_error",
            )
        with self.connect() as db:
            try:
                db.execute(
                    """
                    INSERT INTO identity_providers(
                        id, tenant_id, kind, name, issuer, client_id,
                        authorization_endpoint, token_endpoint, jwks_uri,
                        client_secret_env, service_token_auth_method,
                        private_key_env, private_key_kid, deployment_id, default_role,
                        enabled, created_by, created_at
                    ) VALUES (
                        :id, :tenant_id, :kind, :name, :issuer, :client_id,
                        :authorization_endpoint, :token_endpoint, :jwks_uri,
                        :client_secret_env, :service_token_auth_method,
                        :private_key_env, :private_key_kid, :deployment_id, :default_role,
                        :enabled, :created_by, :created_at
                    )
                    """,
                    provider,
                )
            except DatabaseIntegrityError:
                raise ApiError(409, "Identity provider is already configured", "conflict")
            self.record_audit(
                db,
                "identity_provider.created",
                auth["user"]["tenantId"],
                auth["user"]["id"],
                "identity_provider",
                provider["id"],
                {"kind": kind, "issuer": provider["issuer"]},
                auth.get("_request"),
            )
        return self.identity_provider_payload(provider, include_private=True)

    def identity_provider_payload(self, row, include_private=False):
        data = row_dict(row)
        payload = {
            "id": data["id"],
            "kind": data["kind"],
            "name": data["name"],
            "issuer": data["issuer"],
            "clientId": data["client_id"],
            "authorizationEndpoint": data["authorization_endpoint"],
            "tokenEndpoint": data["token_endpoint"],
            "jwksUri": data["jwks_uri"],
            "deploymentId": data["deployment_id"],
            "serviceTokenAuthMethod": data.get("service_token_auth_method")
            or "client_secret_basic",
            "privateKeyKid": data.get("private_key_kid"),
            "defaultRole": data["default_role"],
            "enabled": bool(data["enabled"]),
            "createdAt": data["created_at"],
        }
        if include_private:
            payload["clientSecretEnv"] = data["client_secret_env"]
            payload["privateKeyEnv"] = data.get("private_key_env")
        return payload

    def update_identity_provider(self, auth, provider_id, payload):
        self.require_role(auth, "admin")
        self.require_verified(auth)
        with self.connect() as db:
            current = db.execute(
                """
                SELECT * FROM identity_providers
                WHERE id = ? AND tenant_id = ?
                """,
                (provider_id, auth["user"]["tenantId"]),
            ).fetchone()
            if not current:
                raise ApiError(404, "Identity provider was not found", "not_found")
            current = row_dict(current)
            field_map = {
                "name": "name",
                "issuer": "issuer",
                "clientId": "client_id",
                "authorizationEndpoint": "authorization_endpoint",
                "tokenEndpoint": "token_endpoint",
                "jwksUri": "jwks_uri",
                "clientSecretEnv": "client_secret_env",
                "serviceTokenAuthMethod": "service_token_auth_method",
                "privateKeyEnv": "private_key_env",
                "privateKeyKid": "private_key_kid",
                "deploymentId": "deployment_id",
                "defaultRole": "default_role",
                "enabled": "enabled",
            }
            values = dict(current)
            for source, target in field_map.items():
                if source in payload:
                    values[target] = payload[source]
            values["name"] = required_text(values["name"], "Provider name", 100)
            values["issuer"] = required_text(
                values["issuer"],
                "Issuer",
                500,
            ).rstrip("/")
            values["client_id"] = required_text(
                values["client_id"],
                "Client ID",
                300,
            )
            values["authorization_endpoint"] = required_text(
                values["authorization_endpoint"],
                "Authorization endpoint",
                1000,
            )
            values["token_endpoint"] = (
                str(values.get("token_endpoint") or "").strip() or None
            )
            values["jwks_uri"] = required_text(
                values["jwks_uri"],
                "JWKS URI",
                1000,
            )
            for key, label in (
                ("client_secret_env", "Client secret"),
                ("private_key_env", "Private key"),
            ):
                values[key] = str(values.get(key) or "").strip() or None
                if values[key] and not re.fullmatch(
                    r"[A-Z][A-Z0-9_]{2,80}",
                    values[key],
                ):
                    raise ApiError(
                        400,
                        f"{label} environment variable name is invalid",
                        "validation_error",
                    )
            values["service_token_auth_method"] = str(
                values.get("service_token_auth_method") or "client_secret_basic"
            )
            if values["service_token_auth_method"] not in (
                "client_secret_basic",
                "private_key_jwt",
            ):
                raise ApiError(
                    400,
                    "LTI service token authentication method is invalid",
                    "validation_error",
                )
            values["private_key_kid"] = (
                str(values.get("private_key_kid") or "").strip() or None
            )
            values["deployment_id"] = (
                str(values.get("deployment_id") or "").strip() or None
            )
            values["default_role"] = str(values.get("default_role") or "student")
            if values["default_role"] not in ("professor", "student"):
                raise ApiError(400, "Default role is invalid", "validation_error")
            values["enabled"] = 1 if bool(values.get("enabled")) else 0
            if values["kind"] == "oidc" and not values["token_endpoint"]:
                raise ApiError(400, "OIDC token endpoint is required", "validation_error")
            if values["kind"] == "lti" and not values["deployment_id"]:
                raise ApiError(400, "LTI deployment ID is required", "validation_error")
            if (
                values["kind"] == "lti"
                and values["service_token_auth_method"] == "private_key_jwt"
                and not values["private_key_env"]
            ):
                raise ApiError(
                    400,
                    "Private key environment variable is required",
                    "validation_error",
                )
            try:
                db.execute(
                    """
                    UPDATE identity_providers
                    SET name = :name,
                        issuer = :issuer,
                        client_id = :client_id,
                        authorization_endpoint = :authorization_endpoint,
                        token_endpoint = :token_endpoint,
                        jwks_uri = :jwks_uri,
                        client_secret_env = :client_secret_env,
                        service_token_auth_method = :service_token_auth_method,
                        private_key_env = :private_key_env,
                        private_key_kid = :private_key_kid,
                        deployment_id = :deployment_id,
                        default_role = :default_role,
                        enabled = :enabled
                    WHERE id = :id AND tenant_id = :tenant_id
                    """,
                    values,
                )
            except DatabaseIntegrityError:
                raise ApiError(409, "Identity provider is already configured", "conflict")
            self.record_audit(
                db,
                "identity_provider.updated",
                auth["user"]["tenantId"],
                auth["user"]["id"],
                "identity_provider",
                provider_id,
                {
                    "kind": values["kind"],
                    "enabled": bool(values["enabled"]),
                    "tokenAuthMethod": values["service_token_auth_method"],
                },
                auth.get("_request"),
            )
            updated = db.execute(
                "SELECT * FROM identity_providers WHERE id = ?",
                (provider_id,),
            ).fetchone()
        return self.identity_provider_payload(updated, include_private=True)

    def list_identity_providers(self, auth):
        self.require_role(auth, "admin")
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT * FROM identity_providers
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                """,
                (auth["user"]["tenantId"],),
            ).fetchall()
            return [
                self.identity_provider_payload(row, include_private=True)
                for row in rows
            ]

    def public_identity_providers(self, tenant_slug):
        tenant_slug = required_text(tenant_slug, "Tenant slug", 40).lower()
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT p.*
                FROM identity_providers p
                JOIN tenants t ON t.id = p.tenant_id
                WHERE t.slug = ? AND p.enabled = 1
                ORDER BY p.name
                """,
                (tenant_slug,),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "name": row["name"],
                    "tenantSlug": tenant_slug,
                }
                for row in rows
            ]

    def get_identity_provider(
        self,
        provider_id=None,
        tenant_slug=None,
        issuer=None,
        client_id=None,
        kind=None,
    ):
        filters = ["p.enabled = 1"]
        parameters = []
        if provider_id:
            filters.append("p.id = ?")
            parameters.append(provider_id)
        if tenant_slug:
            filters.append("t.slug = ?")
            parameters.append(tenant_slug)
        if issuer:
            filters.append("p.issuer = ?")
            parameters.append(str(issuer).rstrip("/"))
        if client_id:
            filters.append("p.client_id = ?")
            parameters.append(client_id)
        if kind:
            filters.append("p.kind = ?")
            parameters.append(kind)
        with self.connect() as db:
            row = db.execute(
                f"""
                SELECT p.*, t.slug AS tenant_slug, t.name AS tenant_name
                FROM identity_providers p
                JOIN tenants t ON t.id = p.tenant_id
                WHERE {" AND ".join(filters)}
                LIMIT 1
                """,
                parameters,
            ).fetchone()
            if not row:
                raise ApiError(404, "Identity provider was not found", "not_found")
            return row_dict(row)

    def create_federation_state(
        self,
        provider,
        kind,
        target_path="/",
        context=None,
    ):
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(24)
        now = utc_now()
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO federation_states(
                    state_hash, tenant_id, provider_id, kind, nonce,
                    target_path, context_json, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_hash(state),
                    provider["tenant_id"],
                    provider["id"],
                    kind,
                    nonce,
                    str(target_path or "/")[:1000],
                    compact_json(context or {}),
                    iso_time(now + timedelta(minutes=10)),
                    iso_time(now),
                ),
            )
        return {"state": state, "nonce": nonce}

    def consume_federation_state(self, state, kind):
        state = required_text(state, "Federation state", 300)
        with self.connect() as db:
            row = db.execute(
                """
                SELECT * FROM federation_states
                WHERE state_hash = ?
                """,
                (session_hash(state),),
            ).fetchone()
            if not row or row["kind"] != kind:
                raise ApiError(400, "Federation state is invalid", "invalid_state")
            expires_at = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
            if expires_at <= utc_now():
                db.execute(
                    "DELETE FROM federation_states WHERE state_hash = ?",
                    (session_hash(state),),
                )
                raise ApiError(400, "Federation state has expired", "expired_state")
            db.execute(
                "DELETE FROM federation_states WHERE state_hash = ?",
                (session_hash(state),),
            )
            result = {
                "nonce": row["nonce"],
                "targetPath": row["target_path"],
                "context": parse_json(row["context_json"], {}),
                "providerId": row["provider_id"],
                "tenantId": row["tenant_id"],
            }
        result["provider"] = self.get_identity_provider(provider_id=result["providerId"])
        return result

    def resolve_federated_login(self, provider, claims, request_context=None):
        subject = required_text(claims.get("sub"), "Identity subject", 500)
        now = iso_time()
        with self.connect() as db:
            identity = db.execute(
                """
                SELECT u.*
                FROM external_identities e
                JOIN users u ON u.id = e.user_id
                WHERE e.provider_id = ? AND e.subject = ?
                """,
                (provider["id"], subject),
            ).fetchone()
            if identity:
                user_id = identity["id"]
                role = identity["role"]
            else:
                role = provider["default_role"]
                lti_roles = claims.get(
                    "https://purl.imsglobal.org/spec/lti/claim/roles",
                    [],
                )
                if any("Instructor" in str(item) for item in lti_roles):
                    role = "professor"
                email = str(claims.get("email") or "").strip().lower()
                if not EMAIL_PATTERN.match(email):
                    digest = hashlib.sha256(subject.encode("utf-8")).hexdigest()[:18]
                    email = f"federated-{digest}@{provider['tenant_slug']}.invalid"
                display_name = str(
                    claims.get("name")
                    or claims.get("preferred_username")
                    or email.split("@", 1)[0]
                )[:80]
                existing = db.execute(
                    "SELECT * FROM users WHERE email = ?",
                    (email,),
                ).fetchone()
                if existing and existing["tenant_id"] != provider["tenant_id"]:
                    raise ApiError(
                        409,
                        "Federated email belongs to another tenant",
                        "conflict",
                    )
                if existing:
                    user_id = existing["id"]
                    role = existing["role"]
                else:
                    user_id = new_id("user")
                    password_salt, password_hash = password_digest(
                        secrets.token_urlsafe(48)
                    )
                    db.execute(
                        """
                        INSERT INTO users(
                            id, tenant_id, email, display_name, role,
                            password_salt, password_hash, email_verified_at,
                            auth_provider, external_subject, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            provider["tenant_id"],
                            email,
                            display_name,
                            role,
                            password_salt,
                            password_hash,
                            now,
                            provider["kind"],
                            subject,
                            now,
                        ),
                    )
                db.execute(
                    """
                    INSERT INTO external_identities(
                        provider_id, subject, user_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(provider_id, subject) DO NOTHING
                    """,
                    (provider["id"], subject, user_id, now),
                )

            course_id = None
            if provider["kind"] == "lti":
                context_claim = claims.get(
                    "https://purl.imsglobal.org/spec/lti/claim/context",
                    {},
                )
                context_id = str(context_claim.get("id") or "").strip()
                if context_id:
                    mapping = db.execute(
                        """
                        SELECT course_id FROM lti_contexts
                        WHERE provider_id = ? AND context_id = ?
                        """,
                        (provider["id"], context_id),
                    ).fetchone()
                    if mapping:
                        course_id = mapping["course_id"]
                    else:
                        owner = db.execute(
                            """
                            SELECT id FROM users
                            WHERE tenant_id = ? AND role = 'admin'
                            ORDER BY created_at LIMIT 1
                            """,
                            (provider["tenant_id"],),
                        ).fetchone()
                        course_id = new_id("course")
                        course_code = (
                            "LTI-" + hashlib.sha256(context_id.encode("utf-8"))
                            .hexdigest()[:8]
                        ).upper()
                        db.execute(
                            """
                            INSERT INTO courses(
                                id, tenant_id, owner_id, name, code,
                                term, join_code, created_at
                            ) VALUES (?, ?, ?, ?, ?, 'LTI', ?, ?)
                            """,
                            (
                                course_id,
                                provider["tenant_id"],
                                owner["id"] if owner else user_id,
                                str(
                                    context_claim.get("title")
                                    or context_claim.get("label")
                                    or "LTI Course"
                                )[:120],
                                course_code,
                                join_code(10),
                                now,
                            ),
                        )
                        db.execute(
                            """
                            INSERT INTO lti_contexts(
                                provider_id, context_id, course_id, created_at
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (provider["id"], context_id, course_id, now),
                        )
                    nrps_claim = claims.get(
                        "https://purl.imsglobal.org/spec/lti-nrps/claim/namesroleservice",
                        {},
                    )
                    ags_claim = claims.get(
                        "https://purl.imsglobal.org/spec/lti-ags/claim/endpoint",
                        {},
                    )
                    resource_link_claim = claims.get(
                        "https://purl.imsglobal.org/spec/lti/claim/resource_link",
                        {},
                    )
                    stored_context = db.execute(
                        """
                        SELECT * FROM lti_contexts
                        WHERE provider_id = ? AND context_id = ?
                        """,
                        (provider["id"], context_id),
                    ).fetchone()
                    has_nrps_claim = (
                        "https://purl.imsglobal.org/spec/lti-nrps/claim/namesroleservice"
                        in claims
                    )
                    has_ags_claim = (
                        "https://purl.imsglobal.org/spec/lti-ags/claim/endpoint"
                        in claims
                    )
                    has_resource_link_claim = (
                        "https://purl.imsglobal.org/spec/lti/claim/resource_link"
                        in claims
                    )
                    db.execute(
                        """
                        UPDATE lti_contexts
                        SET resource_link_id = ?,
                            nrps_memberships_url = ?,
                            nrps_scope_json = ?,
                            ags_lineitems_url = ?,
                            ags_lineitem_url = ?,
                            ags_scope_json = ?
                        WHERE provider_id = ? AND context_id = ?
                        """,
                        (
                            (
                                str(resource_link_claim.get("id") or "").strip()
                                or None
                                if has_resource_link_claim
                                else stored_context["resource_link_id"]
                            ),
                            (
                                str(
                                    nrps_claim.get("context_memberships_url") or ""
                                ).strip()
                                or None
                                if has_nrps_claim
                                else stored_context["nrps_memberships_url"]
                            ),
                            (
                                compact_json(
                                    [
                                        "https://purl.imsglobal.org/spec/lti-nrps/scope/contextmembership.readonly"
                                    ]
                                    if nrps_claim.get("context_memberships_url")
                                    else []
                                )
                                if has_nrps_claim
                                else stored_context["nrps_scope_json"]
                            ),
                            (
                                str(ags_claim.get("lineitems") or "").strip()
                                or None
                                if has_ags_claim
                                else stored_context["ags_lineitems_url"]
                            ),
                            (
                                str(ags_claim.get("lineitem") or "").strip()
                                or None
                                if has_ags_claim
                                else stored_context["ags_lineitem_url"]
                            ),
                            (
                                compact_json(ags_claim.get("scope") or [])
                                if has_ags_claim
                                else stored_context["ags_scope_json"]
                            ),
                            provider["id"],
                            context_id,
                        ),
                    )
                    enrollment_role = "instructor" if role in ("admin", "professor") else "student"
                    db.execute(
                        """
                        INSERT INTO enrollments(course_id, user_id, role, created_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(course_id, user_id) DO UPDATE SET role = excluded.role
                        """,
                        (course_id, user_id, enrollment_role, now),
                    )

            session_token, csrf = self.create_session(db, user_id)
            self.record_audit(
                db,
                "user.federated_login",
                provider["tenant_id"],
                user_id,
                "identity_provider",
                provider["id"],
                {"kind": provider["kind"], "courseId": course_id},
                request_context,
            )
        auth = self.authenticate(session_token)
        auth["csrfToken"] = csrf
        auth["sessionToken"] = session_token
        auth["courseId"] = course_id
        return auth

    def get_lti_course_service(self, auth, course_id, include_private=False):
        self.require_role(auth, "admin", "professor")
        self.require_verified(auth)
        with self.connect() as db:
            self.course_access(db, auth, course_id, instructor=True)
            context = db.execute(
                """
                SELECT *
                FROM lti_contexts
                WHERE course_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (course_id,),
            ).fetchone()
            if not context:
                return {
                    "connected": False,
                    "courseId": course_id,
                    "nrps": {"available": False},
                    "ags": {
                        "available": False,
                        "canCreateLineItems": False,
                        "canSendScores": False,
                    },
                }
            provider = db.execute(
                """
                SELECT * FROM identity_providers
                WHERE id = ? AND tenant_id = ?
                """,
                (context["provider_id"], auth["user"]["tenantId"]),
            ).fetchone()
            if not provider:
                raise ApiError(404, "LTI provider was not found", "not_found")
            provider_data = row_dict(provider)
            context_data = row_dict(context)
        ags_scopes = parse_json(context_data.get("ags_scope_json"), [])
        nrps_scopes = parse_json(context_data.get("nrps_scope_json"), [])
        payload = {
            "connected": True,
            "courseId": course_id,
            "provider": self.identity_provider_payload(
                provider_data,
                include_private=include_private,
            ),
            "contextId": context_data["context_id"],
            "resourceLinkId": context_data.get("resource_link_id"),
            "lastRosterSyncAt": context_data.get("last_roster_sync_at"),
            "nrps": {
                "available": bool(context_data.get("nrps_memberships_url")),
                "membershipsUrl": context_data.get("nrps_memberships_url"),
                "scopes": nrps_scopes,
            },
            "ags": {
                "available": bool(
                    context_data.get("ags_lineitems_url")
                    or context_data.get("ags_lineitem_url")
                ),
                "lineitemsUrl": context_data.get("ags_lineitems_url"),
                "lineitemUrl": context_data.get("ags_lineitem_url"),
                "scopes": ags_scopes,
                "canCreateLineItems": (
                    "https://purl.imsglobal.org/spec/lti-ags/scope/lineitem"
                    in ags_scopes
                    and bool(context_data.get("ags_lineitems_url"))
                ),
                "canSendScores": (
                    "https://purl.imsglobal.org/spec/lti-ags/scope/score"
                    in ags_scopes
                ),
            },
        }
        if include_private:
            payload["_provider"] = provider_data
        return payload

    def apply_lti_roster(self, auth, course_id, provider_id, context_id, members):
        self.require_role(auth, "admin", "professor")
        self.require_verified(auth)
        if not isinstance(members, list):
            raise ApiError(400, "NRPS members must be an array", "validation_error")
        if len(members) > 10000:
            raise ApiError(413, "NRPS roster is too large", "payload_too_large")
        now = iso_time()
        counts = {
            "received": len(members),
            "created": 0,
            "enrolled": 0,
            "inactive": 0,
            "missing": 0,
        }
        seen_subjects = set()
        with self.connect() as db:
            course = self.course_access(db, auth, course_id, instructor=True)
            context = db.execute(
                """
                SELECT * FROM lti_contexts
                WHERE provider_id = ? AND context_id = ? AND course_id = ?
                """,
                (provider_id, context_id, course_id),
            ).fetchone()
            if not context:
                raise ApiError(404, "LTI course context was not found", "not_found")
            provider = db.execute(
                """
                SELECT p.*, t.slug AS tenant_slug
                FROM identity_providers p
                JOIN tenants t ON t.id = p.tenant_id
                WHERE p.id = ? AND p.tenant_id = ? AND p.enabled = 1
                """,
                (provider_id, auth["user"]["tenantId"]),
            ).fetchone()
            if not provider:
                raise ApiError(404, "Active LTI provider was not found", "not_found")
            for member in members:
                if not isinstance(member, dict):
                    continue
                subject = str(member.get("user_id") or "").strip()
                if not subject or len(subject) > 500:
                    continue
                seen_subjects.add(subject)
                status = str(member.get("status") or "Active")[:40]
                roles = member.get("roles") if isinstance(member.get("roles"), list) else []
                instructor = any(
                    marker in str(role)
                    for role in roles
                    for marker in ("Instructor", "Administrator", "TeachingAssistant")
                )
                identity = db.execute(
                    """
                    SELECT u.*
                    FROM external_identities e
                    JOIN users u ON u.id = e.user_id
                    WHERE e.provider_id = ? AND e.subject = ?
                    """,
                    (provider_id, subject),
                ).fetchone()
                if identity:
                    user_id = identity["id"]
                    if instructor and identity["role"] == "student":
                        db.execute(
                            "UPDATE users SET role = 'professor' WHERE id = ?",
                            (user_id,),
                        )
                else:
                    email = str(member.get("email") or "").strip().lower()
                    if not EMAIL_PATTERN.match(email):
                        digest = hashlib.sha256(subject.encode("utf-8")).hexdigest()[:18]
                        email = f"federated-{digest}@{provider['tenant_slug']}.invalid"
                    existing = db.execute(
                        "SELECT * FROM users WHERE email = ?",
                        (email,),
                    ).fetchone()
                    if existing and existing["tenant_id"] != provider["tenant_id"]:
                        digest = hashlib.sha256(
                            f"{provider_id}:{subject}".encode("utf-8")
                        ).hexdigest()[:18]
                        email = f"federated-{digest}@{provider['tenant_slug']}.invalid"
                        existing = db.execute(
                            "SELECT * FROM users WHERE email = ?",
                            (email,),
                        ).fetchone()
                    if existing:
                        user_id = existing["id"]
                    else:
                        user_id = new_id("user")
                        password_salt, password_hash = password_digest(
                            secrets.token_urlsafe(48)
                        )
                        display_name = str(
                            member.get("name")
                            or member.get("given_name")
                            or email.split("@", 1)[0]
                        )[:80]
                        db.execute(
                            """
                            INSERT INTO users(
                                id, tenant_id, email, display_name, role,
                                password_salt, password_hash, email_verified_at,
                                auth_provider, external_subject, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'lti', ?, ?)
                            """,
                            (
                                user_id,
                                provider["tenant_id"],
                                email,
                                display_name,
                                "professor" if instructor else "student",
                                password_salt,
                                password_hash,
                                now,
                                subject,
                                now,
                            ),
                        )
                        counts["created"] += 1
                    db.execute(
                        """
                        INSERT INTO external_identities(
                            provider_id, subject, user_id, created_at
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT(provider_id, subject) DO UPDATE SET user_id = excluded.user_id
                        """,
                        (provider_id, subject, user_id, now),
                    )
                synced_name = str(
                    member.get("name")
                    or " ".join(filter(None, [
                        str(member.get("given_name") or "").strip(),
                        str(member.get("family_name") or "").strip(),
                    ]))
                ).strip()[:80]
                if synced_name:
                    db.execute(
                        "UPDATE users SET display_name = ? WHERE id = ?",
                        (synced_name, user_id),
                    )
                if instructor:
                    db.execute(
                        """
                        UPDATE users
                        SET role = 'professor'
                        WHERE id = ? AND role = 'student'
                        """,
                        (user_id,),
                    )
                db.execute(
                    """
                    INSERT INTO lti_memberships(
                        provider_id, context_id, subject, user_id,
                        status, roles_json, synced_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider_id, context_id, subject) DO UPDATE SET
                        user_id = excluded.user_id,
                        status = excluded.status,
                        roles_json = excluded.roles_json,
                        synced_at = excluded.synced_at
                    """,
                    (
                        provider_id,
                        context_id,
                        subject,
                        user_id,
                        status,
                        compact_json(roles),
                        now,
                    ),
                )
                if status.lower() == "active":
                    db.execute(
                        """
                        INSERT INTO enrollments(course_id, user_id, role, created_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(course_id, user_id) DO UPDATE SET role = excluded.role
                        """,
                        (
                            course_id,
                            user_id,
                            "instructor" if instructor else "student",
                            now,
                        ),
                    )
                    counts["enrolled"] += 1
                else:
                    if user_id != course["owner_id"]:
                        db.execute(
                            "DELETE FROM enrollments WHERE course_id = ? AND user_id = ?",
                            (course_id, user_id),
                        )
                    counts["inactive"] += 1

            previous = db.execute(
                """
                SELECT subject, user_id
                FROM lti_memberships
                WHERE provider_id = ? AND context_id = ?
                """,
                (provider_id, context_id),
            ).fetchall()
            for membership in previous:
                if membership["subject"] in seen_subjects:
                    continue
                db.execute(
                    """
                    UPDATE lti_memberships
                    SET status = 'Missing', synced_at = ?
                    WHERE provider_id = ? AND context_id = ? AND subject = ?
                    """,
                    (now, provider_id, context_id, membership["subject"]),
                )
                if membership["user_id"] != course["owner_id"]:
                    db.execute(
                        "DELETE FROM enrollments WHERE course_id = ? AND user_id = ?",
                        (course_id, membership["user_id"]),
                    )
                counts["missing"] += 1
            db.execute(
                """
                UPDATE lti_contexts
                SET last_roster_sync_at = ?
                WHERE provider_id = ? AND context_id = ?
                """,
                (now, provider_id, context_id),
            )
            self.record_audit(
                db,
                "lti.roster_synced",
                auth["user"]["tenantId"],
                auth["user"]["id"],
                "course",
                course_id,
                counts,
                auth.get("_request"),
            )
        return {**counts, "syncedAt": now}

    def prepare_lti_grade_passback(self, auth, submission_id):
        self.require_role(auth, "admin", "professor")
        self.require_verified(auth)
        with self.connect() as db:
            submission = db.execute(
                """
                SELECT s.*, a.title AS assignment_title
                FROM submissions s
                JOIN assignments a ON a.id = s.assignment_id
                WHERE s.id = ? AND s.tenant_id = ?
                """,
                (submission_id, auth["user"]["tenantId"]),
            ).fetchone()
            if not submission:
                raise ApiError(404, "Submission was not found", "not_found")
            self.course_access(db, auth, submission["course_id"], instructor=True)
            if submission["status"] != "graded" or submission["score"] is None:
                raise ApiError(
                    409,
                    "Submission must be graded before LMS passback",
                    "submission_not_graded",
                )
            context = db.execute(
                """
                SELECT * FROM lti_contexts
                WHERE course_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (submission["course_id"],),
            ).fetchone()
            if not context:
                raise ApiError(409, "Course is not connected to LTI", "lti_not_connected")
            provider = db.execute(
                """
                SELECT * FROM identity_providers
                WHERE id = ? AND tenant_id = ? AND enabled = 1
                """,
                (context["provider_id"], auth["user"]["tenantId"]),
            ).fetchone()
            if not provider:
                raise ApiError(409, "LTI provider is disabled", "lti_provider_disabled")
            identity = db.execute(
                """
                SELECT subject FROM external_identities
                WHERE provider_id = ? AND user_id = ?
                """,
                (provider["id"], submission["student_id"]),
            ).fetchone()
            if not identity:
                raise ApiError(
                    409,
                    "Student does not have an LMS identity",
                    "lti_student_identity_missing",
                )
            line_item = db.execute(
                """
                SELECT * FROM lti_line_items
                WHERE assignment_id = ?
                """,
                (submission["assignment_id"],),
            ).fetchone()
            return {
                "provider": row_dict(provider),
                "context": row_dict(context),
                "submissionId": submission["id"],
                "assignment": {
                    "id": submission["assignment_id"],
                    "title": submission["assignment_title"],
                },
                "studentSubject": identity["subject"],
                "score": float(submission["score"]),
                "feedback": submission["feedback"],
                "lineitemUrl": (
                    line_item["lineitem_url"]
                    if line_item
                    else context["ags_lineitem_url"]
                ),
            }

    def save_lti_line_item(self, auth, plan, lineitem_url):
        now = iso_time()
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO lti_line_items(
                    assignment_id, provider_id, context_id,
                    lineitem_url, score_maximum, created_at
                ) VALUES (?, ?, ?, ?, 100, ?)
                ON CONFLICT(assignment_id) DO UPDATE SET
                    provider_id = excluded.provider_id,
                    context_id = excluded.context_id,
                    lineitem_url = excluded.lineitem_url,
                    score_maximum = excluded.score_maximum
                """,
                (
                    plan["assignment"]["id"],
                    plan["provider"]["id"],
                    plan["context"]["context_id"],
                    lineitem_url,
                    now,
                ),
            )
            self.record_audit(
                db,
                "lti.lineitem_linked",
                auth["user"]["tenantId"],
                auth["user"]["id"],
                "assignment",
                plan["assignment"]["id"],
                {"lineitemUrl": lineitem_url},
                auth.get("_request"),
            )
        return lineitem_url

    def record_lti_grade_passback(self, auth, plan, lineitem_url, result):
        sent_at = iso_time()
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO lti_grade_passbacks(
                    id, submission_id, provider_id, lineitem_url,
                    score_given, response_json, sent_by, sent_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("passback"),
                    plan["submissionId"],
                    plan["provider"]["id"],
                    lineitem_url,
                    plan["score"],
                    compact_json(result.get("response") or {}),
                    auth["user"]["id"],
                    sent_at,
                ),
            )
            self.record_audit(
                db,
                "lti.grade_passback_sent",
                auth["user"]["tenantId"],
                auth["user"]["id"],
                "submission",
                plan["submissionId"],
                {
                    "score": plan["score"],
                    "lineitemUrl": lineitem_url,
                    "studentSubject": plan["studentSubject"],
                },
                auth.get("_request"),
            )
        return {
            "status": "sent",
            "submissionId": plan["submissionId"],
            "score": plan["score"],
            "sentAt": sent_at,
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
        self.require_verified(auth)
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
            self.record_audit(
                db,
                "assignment.created",
                auth["user"]["tenantId"],
                auth["user"]["id"],
                "assignment",
                assignment["id"],
                {"courseId": course_id, "requiredFamily": required_family},
                auth.get("_request"),
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
        self.require_verified(auth)
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
            self.record_audit(
                db,
                "project.version_saved",
                auth["user"]["tenantId"],
                auth["user"]["id"],
                "project",
                project_id,
                {"courseId": course_id, "versionId": version_id},
                auth.get("_request"),
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
        self.require_verified(auth)
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
            self.record_audit(
                db,
                "submission.created",
                auth["user"]["tenantId"],
                auth["user"]["id"],
                "submission",
                submission["id"],
                {"assignmentId": assignment_id, "attempt": attempt},
                auth.get("_request"),
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
        self.require_verified(auth)
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
            self.record_audit(
                db,
                "submission.graded",
                auth["user"]["tenantId"],
                auth["user"]["id"],
                "submission",
                submission_id,
                {"score": round(score, 1)},
                auth.get("_request"),
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
