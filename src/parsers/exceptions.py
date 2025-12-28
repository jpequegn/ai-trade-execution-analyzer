"""Custom exceptions for FIX protocol parsing."""


class FIXParseError(Exception):
    """Base exception for FIX parsing errors.

    Raised when a FIX message cannot be parsed due to format issues.
    """

    def __init__(self, message: str, raw_message: str | None = None) -> None:
        """Initialize FIXParseError.

        Args:
            message: Human-readable error description.
            raw_message: The raw FIX message that failed to parse (optional).
        """
        super().__init__(message)
        self.raw_message = raw_message


class FIXValidationError(FIXParseError):
    """Raised when a FIX message fails validation.

    This includes invalid field values, type mismatches, or constraint violations.
    """

    def __init__(
        self,
        message: str,
        field_tag: int | None = None,
        field_value: str | None = None,
        raw_message: str | None = None,
    ) -> None:
        """Initialize FIXValidationError.

        Args:
            message: Human-readable error description.
            field_tag: The FIX tag number that failed validation (optional).
            field_value: The value that failed validation (optional).
            raw_message: The raw FIX message that failed to parse (optional).
        """
        super().__init__(message, raw_message)
        self.field_tag = field_tag
        self.field_value = field_value


class FIXMissingFieldError(FIXParseError):
    """Raised when a required FIX field is missing.

    This exception is raised when a mandatory field is not present in the message.
    """

    def __init__(
        self,
        field_tag: int,
        field_name: str | None = None,
        raw_message: str | None = None,
    ) -> None:
        """Initialize FIXMissingFieldError.

        Args:
            field_tag: The FIX tag number of the missing field.
            field_name: Human-readable name of the field (optional).
            raw_message: The raw FIX message that failed to parse (optional).
        """
        name_part = f" ({field_name})" if field_name else ""
        message = f"Required FIX field {field_tag}{name_part} is missing"
        super().__init__(message, raw_message)
        self.field_tag = field_tag
        self.field_name = field_name
