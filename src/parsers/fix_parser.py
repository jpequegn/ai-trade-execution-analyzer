"""FIX protocol message parser for execution reports.

This module provides functionality to parse FIX 4.2/4.4 execution report
messages into structured Python objects for analysis.

FIX (Financial Information eXchange) protocol is the standard for electronic
trading communication. This parser focuses on ExecutionReport (MsgType=8)
messages which contain trade execution details.

Example:
    >>> from src.parsers.fix_parser import parse_fix_message
    >>> msg = "8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|32=100|31=150.50|30=NYSE|60=20240115-10:30:00.000|39=2"
    >>> report = parse_fix_message(msg)
    >>> print(report.symbol)
    AAPL
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.parsers.exceptions import (
    FIXMissingFieldError,
    FIXParseError,
    FIXValidationError,
)

# FIX Tag Constants
TAG_BEGIN_STRING = 8  # FIX version (e.g., "FIX.4.4")
TAG_MSG_TYPE = 35  # Message type (8 = Execution Report)
TAG_ORDER_ID = 37  # Order ID
TAG_SYMBOL = 55  # Trading symbol
TAG_SIDE = 54  # Side (1=Buy, 2=Sell)
TAG_LAST_QTY = 32  # Last fill quantity
TAG_LAST_PX = 31  # Last fill price
TAG_LAST_MKT = 30  # Last market/venue
TAG_TRANSACT_TIME = 60  # Transaction timestamp
TAG_ORD_STATUS = 39  # Order status
TAG_EXEC_TYPE = 150  # Execution type
TAG_CUM_QTY = 14  # Cumulative quantity
TAG_AVG_PX = 6  # Average price

# Side mappings
SIDE_BUY = "1"
SIDE_SELL = "2"

# Order status mappings (relevant for fill type)
ORD_STATUS_FILLED = "2"  # Full fill
ORD_STATUS_PARTIAL = "1"  # Partial fill

# SOH delimiter (ASCII 01) and common test delimiter
SOH = "\x01"
PIPE = "|"

# Timestamp formats supported
TIMESTAMP_FORMATS = [
    "%Y%m%d-%H:%M:%S.%f",  # Standard with milliseconds
    "%Y%m%d-%H:%M:%S",  # Without milliseconds
    "%Y%m%d %H:%M:%S.%f",  # Space separator with milliseconds
    "%Y%m%d %H:%M:%S",  # Space separator without milliseconds
]


class ExecutionReport(BaseModel):
    """Structured representation of a FIX ExecutionReport message.

    This model represents the key fields from a FIX ExecutionReport (MsgType=8)
    that are relevant for trade execution analysis.

    Attributes:
        order_id: Unique identifier for the order.
        symbol: Trading symbol (e.g., "AAPL", "GOOGL").
        side: Trade direction - "BUY" or "SELL".
        quantity: Number of shares/units filled in this execution.
        price: Execution price per share/unit.
        venue: Execution venue/exchange (e.g., "NYSE", "NASDAQ").
        timestamp: Time of execution.
        fill_type: "FULL" for complete fill, "PARTIAL" for partial fill.
        exec_type: Execution type code (optional).
        cum_qty: Cumulative quantity filled (optional).
        avg_px: Average fill price (optional).
        fix_version: FIX protocol version (e.g., "FIX.4.4").
    """

    order_id: str = Field(..., min_length=1, description="Unique order identifier")
    symbol: str = Field(..., min_length=1, description="Trading symbol")
    side: Literal["BUY", "SELL"] = Field(..., description="Trade direction")
    quantity: float = Field(..., gt=0, description="Fill quantity")
    price: float = Field(..., gt=0, description="Fill price")
    venue: str = Field(default="UNKNOWN", description="Execution venue")
    timestamp: datetime = Field(..., description="Execution timestamp")
    fill_type: Literal["FULL", "PARTIAL"] = Field(..., description="Fill completeness")
    exec_type: str | None = Field(default=None, description="Execution type code")
    cum_qty: float | None = Field(default=None, ge=0, description="Cumulative quantity")
    avg_px: float | None = Field(default=None, ge=0, description="Average price")
    fix_version: str = Field(default="FIX.4.4", description="FIX protocol version")

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()


def tokenize_fix_message(raw_message: str) -> dict[int, str]:
    """Tokenize a FIX message into tag-value pairs.

    Splits a FIX message by the delimiter (SOH or pipe) and extracts
    tag-value pairs.

    Args:
        raw_message: Raw FIX message string.

    Returns:
        Dictionary mapping tag numbers to their string values.

    Raises:
        FIXParseError: If the message format is invalid.

    Example:
        >>> tokenize_fix_message("8=FIX.4.4|35=8|55=AAPL")
        {8: 'FIX.4.4', 35: '8', 55: 'AAPL'}
    """
    if not raw_message or not raw_message.strip():
        raise FIXParseError("Empty FIX message", raw_message)

    # Determine delimiter (SOH or pipe)
    delimiter = SOH if SOH in raw_message else PIPE

    # Split and parse fields
    fields: dict[int, str] = {}
    pairs = raw_message.split(delimiter)

    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue

        if "=" not in pair:
            # Skip invalid pairs silently (lenient parsing)
            continue

        try:
            tag_str, value = pair.split("=", 1)
            tag = int(tag_str)
            fields[tag] = value
        except ValueError:
            # Skip pairs with non-numeric tags
            continue

    if not fields:
        raise FIXParseError("No valid tag-value pairs found", raw_message)

    return fields


def parse_timestamp(value: str) -> datetime:
    """Parse a FIX timestamp string into a datetime object.

    Supports multiple timestamp formats commonly used in FIX messages.

    Args:
        value: Timestamp string in FIX format.

    Returns:
        Parsed datetime object.

    Raises:
        FIXValidationError: If timestamp format is not recognized.
    """
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise FIXValidationError(
        f"Invalid timestamp format: {value}",
        field_tag=TAG_TRANSACT_TIME,
        field_value=value,
    )


def parse_float(value: str, tag: int, field_name: str) -> float:
    """Parse a string value to float with validation.

    Args:
        value: String value to parse.
        tag: FIX tag number (for error context).
        field_name: Human-readable field name (for error context).

    Returns:
        Parsed float value.

    Raises:
        FIXValidationError: If value cannot be parsed as float.
    """
    try:
        return float(value)
    except ValueError as e:
        raise FIXValidationError(
            f"Invalid {field_name} value: {value}",
            field_tag=tag,
            field_value=value,
        ) from e


def parse_side(value: str) -> Literal["BUY", "SELL"]:
    """Parse FIX side code to human-readable string.

    Args:
        value: FIX side code ("1" for Buy, "2" for Sell).

    Returns:
        "BUY" or "SELL".

    Raises:
        FIXValidationError: If side code is not recognized.
    """
    if value == SIDE_BUY:
        return "BUY"
    if value == SIDE_SELL:
        return "SELL"

    raise FIXValidationError(
        f"Invalid side value: {value} (expected '1' for Buy or '2' for Sell)",
        field_tag=TAG_SIDE,
        field_value=value,
    )


def parse_fill_type(value: str) -> Literal["FULL", "PARTIAL"]:
    """Parse FIX order status to fill type.

    Args:
        value: FIX order status code.

    Returns:
        "FULL" for complete fill, "PARTIAL" for partial fill.
    """
    if value == ORD_STATUS_FILLED:
        return "FULL"
    return "PARTIAL"


def get_required_field(
    fields: dict[int, str],
    tag: int,
    field_name: str,
    raw_message: str | None = None,
) -> str:
    """Get a required field value from parsed fields.

    Args:
        fields: Dictionary of parsed tag-value pairs.
        tag: FIX tag number to retrieve.
        field_name: Human-readable field name (for error context).
        raw_message: Original raw message (for error context).

    Returns:
        Field value as string.

    Raises:
        FIXMissingFieldError: If the required field is not present.
    """
    if tag not in fields:
        raise FIXMissingFieldError(tag, field_name, raw_message)
    return fields[tag]


def parse_fix_message(raw_message: str) -> ExecutionReport:
    """Parse a FIX execution report message into structured data.

    This function takes a raw FIX message string and extracts the relevant
    fields for trade execution analysis.

    Args:
        raw_message: Raw FIX message string. Can use either SOH (ASCII 01)
            or pipe (|) as field delimiter.

    Returns:
        ExecutionReport object with parsed fields.

    Raises:
        FIXParseError: If the message format is invalid.
        FIXValidationError: If field values are invalid.
        FIXMissingFieldError: If required fields are missing.

    Example:
        >>> msg = "8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|32=100|31=150.50|30=NYSE|60=20240115-10:30:00.000|39=2"
        >>> report = parse_fix_message(msg)
        >>> report.symbol
        'AAPL'
        >>> report.side
        'BUY'
        >>> report.price
        150.5
    """
    # Tokenize the message
    fields = tokenize_fix_message(raw_message)

    # Extract required fields
    order_id = get_required_field(fields, TAG_ORDER_ID, "OrderID", raw_message)
    symbol = get_required_field(fields, TAG_SYMBOL, "Symbol", raw_message)
    side_code = get_required_field(fields, TAG_SIDE, "Side", raw_message)
    qty_str = get_required_field(fields, TAG_LAST_QTY, "LastQty", raw_message)
    px_str = get_required_field(fields, TAG_LAST_PX, "LastPx", raw_message)
    time_str = get_required_field(fields, TAG_TRANSACT_TIME, "TransactTime", raw_message)

    # Parse field values
    side = parse_side(side_code)
    quantity = parse_float(qty_str, TAG_LAST_QTY, "LastQty")
    price = parse_float(px_str, TAG_LAST_PX, "LastPx")
    timestamp = parse_timestamp(time_str)

    # Extract optional fields
    venue = fields.get(TAG_LAST_MKT, "UNKNOWN")
    ord_status = fields.get(TAG_ORD_STATUS, "1")  # Default to partial
    fill_type = parse_fill_type(ord_status)
    fix_version = fields.get(TAG_BEGIN_STRING, "FIX.4.4")
    exec_type = fields.get(TAG_EXEC_TYPE)

    # Parse optional numeric fields
    cum_qty: float | None = None
    avg_px: float | None = None

    if TAG_CUM_QTY in fields:
        cum_qty = parse_float(fields[TAG_CUM_QTY], TAG_CUM_QTY, "CumQty")

    if TAG_AVG_PX in fields:
        avg_px = parse_float(fields[TAG_AVG_PX], TAG_AVG_PX, "AvgPx")

    # Build and return the ExecutionReport
    return ExecutionReport(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        venue=venue,
        timestamp=timestamp,
        fill_type=fill_type,
        exec_type=exec_type,
        cum_qty=cum_qty,
        avg_px=avg_px,
        fix_version=fix_version,
    )
