"""Domain exceptions mapped to stable API errors."""


class PhantomError(Exception):
    """Base class for expected, user-safe failures."""

    code = "phantom_error"


class SessionNotFoundError(PhantomError):
    code = "session_not_found"


class SessionExpiredError(PhantomError):
    code = "session_expired"


class ConsentRequiredError(PhantomError):
    code = "consent_required"


class InvalidMediaError(PhantomError):
    code = "invalid_media"


class PayloadTooLargeError(PhantomError):
    code = "payload_too_large"


class CapacityError(PhantomError):
    code = "session_capacity_reached"
