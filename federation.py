import base64
import hmac
import json
import os
import re
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen


NRPS_MEMBERSHIP_SCOPE = (
    "https://purl.imsglobal.org/spec/lti-nrps/scope/contextmembership.readonly"
)
AGS_LINEITEM_SCOPE = "https://purl.imsglobal.org/spec/lti-ags/scope/lineitem"
AGS_SCORE_SCOPE = "https://purl.imsglobal.org/spec/lti-ags/scope/score"


class FederationError(RuntimeError):
    def __init__(self, message, code="federation_error", status=400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


def require_pyjwt():
    try:
        import jwt
    except ImportError as error:
        raise FederationError(
            "OIDC/LTI requires PyJWT with crypto support. Install requirements.txt.",
            "federation_dependency_missing",
            503,
        ) from error
    return jwt


def oidc_authorization_url(provider, state, nonce, redirect_uri):
    query = urlencode({
        "client_id": provider["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid profile email",
        "state": state,
        "nonce": nonce,
    })
    separator = "&" if "?" in provider["authorization_endpoint"] else "?"
    return f"{provider['authorization_endpoint']}{separator}{query}"


def lti_authorization_url(
    provider,
    state,
    nonce,
    redirect_uri,
    login_hint,
    lti_message_hint=None,
):
    parameters = {
        "client_id": provider["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "id_token",
        "response_mode": "form_post",
        "scope": "openid",
        "prompt": "none",
        "state": state,
        "nonce": nonce,
        "login_hint": login_hint,
    }
    if lti_message_hint:
        parameters["lti_message_hint"] = lti_message_hint
    query = urlencode(parameters)
    separator = "&" if "?" in provider["authorization_endpoint"] else "?"
    return f"{provider['authorization_endpoint']}{separator}{query}"


def exchange_oidc_code(provider, code, redirect_uri):
    secret_environment = provider.get("client_secret_env")
    client_secret = os.environ.get(secret_environment) if secret_environment else None
    if secret_environment and client_secret is None:
        raise FederationError(
            f"OIDC client secret environment variable is missing: {secret_environment}",
            "oidc_secret_missing",
            503,
        )
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": provider["client_id"],
    }
    if client_secret:
        data["client_secret"] = client_secret
    request = Request(
        provider["token_endpoint"],
        data=urlencode(data).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, json.JSONDecodeError) as error:
        raise FederationError(
            "OIDC token exchange failed",
            "oidc_token_exchange_failed",
            502,
        ) from error
    if not payload.get("id_token"):
        raise FederationError(
            "OIDC token response does not contain id_token",
            "oidc_id_token_missing",
            502,
        )
    return payload


def verify_id_token(provider, id_token, nonce):
    jwt = require_pyjwt()
    try:
        header = jwt.get_unverified_header(id_token)
        algorithm = header.get("alg")
        if algorithm not in ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512"):
            raise FederationError("ID token algorithm is not allowed", "invalid_id_token")
        signing_key = jwt.PyJWKClient(provider["jwks_uri"]).get_signing_key_from_jwt(
            id_token
        ).key
        claims = jwt.decode(
            id_token,
            signing_key,
            algorithms=[algorithm],
            audience=provider["client_id"],
            issuer=provider["issuer"],
            options={
                "require": ["exp", "iat", "iss", "aud", "sub", "nonce"],
            },
        )
    except FederationError:
        raise
    except Exception as error:
        raise FederationError(
            "ID token signature or claims are invalid",
            "invalid_id_token",
        ) from error
    if not hmac.compare_digest(str(claims.get("nonce") or ""), str(nonce or "")):
        raise FederationError("ID token nonce is invalid", "invalid_nonce")
    return claims


def validate_lti_claims(provider, claims):
    deployment_id = claims.get(
        "https://purl.imsglobal.org/spec/lti/claim/deployment_id"
    )
    if not hmac.compare_digest(
        str(deployment_id or ""),
        str(provider.get("deployment_id") or ""),
    ):
        raise FederationError("LTI deployment ID is invalid", "invalid_lti_deployment")
    message_type = claims.get(
        "https://purl.imsglobal.org/spec/lti/claim/message_type"
    )
    if message_type not in (
        "LtiResourceLinkRequest",
        "LtiDeepLinkingRequest",
    ):
        raise FederationError("LTI message type is not supported", "invalid_lti_message")
    version = claims.get("https://purl.imsglobal.org/spec/lti/claim/version")
    if version != "1.3.0":
        raise FederationError("LTI version must be 1.3.0", "invalid_lti_version")
    return claims


def validate_service_url(value):
    value = str(value or "").strip()
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise FederationError("LTI service URL is invalid", "invalid_lti_service_url")
    if parsed.username or parsed.password:
        raise FederationError(
            "LTI service URL must not contain credentials",
            "invalid_lti_service_url",
        )
    if parsed.scheme != "https" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise FederationError(
            "LTI service URL must use HTTPS",
            "invalid_lti_service_url",
        )
    return value


def normalize_scopes(scopes):
    if isinstance(scopes, str):
        values = scopes.split()
    else:
        values = scopes or []
    return sorted({str(value).strip() for value in values if str(value).strip()})


def require_claimed_scopes(claimed_scopes, required_scopes):
    claimed = set(normalize_scopes(claimed_scopes))
    required = set(normalize_scopes(required_scopes))
    missing = sorted(required - claimed)
    if missing:
        raise FederationError(
            f"LTI service scope is unavailable: {', '.join(missing)}",
            "lti_scope_unavailable",
            409,
        )
    return sorted(required)


def _service_secret(provider, environment_key):
    environment_name = str(provider.get(environment_key) or "").strip()
    if not environment_name:
        raise FederationError(
            "LTI service credential environment variable is not configured",
            "lti_service_credential_missing",
            503,
        )
    value = os.environ.get(environment_name)
    if not value:
        raise FederationError(
            f"LTI service credential environment variable is missing: {environment_name}",
            "lti_service_credential_missing",
            503,
        )
    return value


def request_lti_service_token(provider, scopes):
    token_endpoint = validate_service_url(provider.get("token_endpoint"))
    scopes = normalize_scopes(scopes)
    if not scopes:
        raise FederationError("LTI service scope is required", "lti_scope_unavailable")
    auth_method = provider.get("service_token_auth_method") or "client_secret_basic"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "grant_type": "client_credentials",
        "scope": " ".join(scopes),
    }
    if auth_method == "private_key_jwt":
        jwt = require_pyjwt()
        private_key = _service_secret(provider, "private_key_env").replace("\\n", "\n")
        now = int(time.time())
        assertion_headers = {"typ": "JWT"}
        if provider.get("private_key_kid"):
            assertion_headers["kid"] = provider["private_key_kid"]
        assertion = jwt.encode(
            {
                "iss": provider["client_id"],
                "sub": provider["client_id"],
                "aud": token_endpoint,
                "iat": now,
                "exp": now + 300,
                "jti": uuid.uuid4().hex,
            },
            private_key,
            algorithm="RS256",
            headers=assertion_headers,
        )
        data["client_assertion_type"] = (
            "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
        )
        data["client_assertion"] = assertion
    elif auth_method == "client_secret_basic":
        secret = _service_secret(provider, "client_secret_env")
        credential = f"{provider['client_id']}:{secret}".encode("utf-8")
        headers["Authorization"] = (
            "Basic " + base64.b64encode(credential).decode("ascii")
        )
    else:
        raise FederationError(
            "LTI service token authentication method is invalid",
            "lti_token_auth_invalid",
            503,
        )
    request = Request(
        token_endpoint,
        data=urlencode(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, json.JSONDecodeError) as error:
        raise FederationError(
            "LTI service access token request failed",
            "lti_token_request_failed",
            502,
        ) from error
    token = str(payload.get("access_token") or "").strip()
    if not token:
        raise FederationError(
            "LTI service token response does not contain access_token",
            "lti_token_missing",
            502,
        )
    return token


def _request_lti_json(
    url,
    token,
    *,
    method="GET",
    payload=None,
    accept="application/json",
    content_type="application/json",
):
    url = validate_service_url(url)
    data = None
    headers = {
        "Accept": accept,
        "Authorization": f"Bearer {token}",
    }
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = content_type
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read()
            response_payload = json.loads(raw.decode("utf-8")) if raw else {}
            return response_payload, dict(response.headers), response.geturl()
    except (HTTPError, URLError, json.JSONDecodeError) as error:
        raise FederationError(
            "LTI service request failed",
            "lti_service_request_failed",
            502,
        ) from error


def _next_link(headers):
    link_header = headers.get("Link") or headers.get("link") or ""
    for entry in link_header.split(","):
        match = re.match(r'\s*<([^>]+)>\s*;\s*rel="?next"?', entry.strip())
        if match:
            return match.group(1)
    return None


def fetch_nrps_members(provider, memberships_url, claimed_scopes):
    scopes = require_claimed_scopes(
        claimed_scopes,
        [NRPS_MEMBERSHIP_SCOPE],
    )
    token = request_lti_service_token(provider, scopes)
    members = []
    next_url = validate_service_url(memberships_url)
    pages = 0
    while next_url:
        pages += 1
        if pages > 50:
            raise FederationError(
                "NRPS pagination exceeded the safety limit",
                "nrps_pagination_limit",
                502,
            )
        payload, headers, _ = _request_lti_json(
            next_url,
            token,
            accept="application/vnd.ims.lti-nrps.v2.membershipcontainer+json",
        )
        page_members = payload.get("members", [])
        if not isinstance(page_members, list):
            raise FederationError(
                "NRPS response members must be an array",
                "invalid_nrps_response",
                502,
            )
        members.extend(page_members)
        next_url = _next_link(headers)
    return {"members": members, "pages": pages}


def create_ags_line_item(
    provider,
    lineitems_url,
    claimed_scopes,
    assignment,
):
    scopes = require_claimed_scopes(claimed_scopes, [AGS_LINEITEM_SCOPE])
    token = request_lti_service_token(provider, scopes)
    payload = {
        "scoreMaximum": 100,
        "label": str(assignment["title"])[:255],
        "resourceId": assignment["id"],
        "tag": "neural-blocks-lab",
    }
    response, headers, response_url = _request_lti_json(
        lineitems_url,
        token,
        method="POST",
        payload=payload,
        accept="application/vnd.ims.lis.v2.lineitem+json",
        content_type="application/vnd.ims.lis.v2.lineitem+json",
    )
    lineitem_url = (
        response.get("id")
        or headers.get("Location")
        or headers.get("location")
        or response_url
    )
    return {"url": validate_service_url(lineitem_url), "payload": response}


def post_ags_score(
    provider,
    lineitem_url,
    claimed_scopes,
    score,
):
    scopes = require_claimed_scopes(claimed_scopes, [AGS_SCORE_SCOPE])
    token = request_lti_service_token(provider, scopes)
    lineitem = urlsplit(validate_service_url(lineitem_url))
    scores_url = urlunsplit((
        lineitem.scheme,
        lineitem.netloc,
        lineitem.path.rstrip("/") + "/scores",
        lineitem.query,
        "",
    ))
    payload = {
        "userId": score["userId"],
        "scoreGiven": float(score["scoreGiven"]),
        "scoreMaximum": float(score.get("scoreMaximum", 100)),
        "timestamp": score["timestamp"],
        "activityProgress": score.get("activityProgress", "Completed"),
        "gradingProgress": score.get("gradingProgress", "FullyGraded"),
    }
    if score.get("comment"):
        payload["comment"] = str(score["comment"])[:2048]
    response, _, _ = _request_lti_json(
        scores_url,
        token,
        method="POST",
        payload=payload,
        accept="application/json",
        content_type="application/vnd.ims.lis.v1.score+json",
    )
    return {"status": "sent", "payload": payload, "response": response}
