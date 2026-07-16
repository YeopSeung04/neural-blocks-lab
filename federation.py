import hmac
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


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
