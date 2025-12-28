"""Comprehensive tests for FIX protocol parser."""

from datetime import datetime

import pytest

from src.parsers import (
    ExecutionReport,
    FIXMissingFieldError,
    FIXParseError,
    FIXValidationError,
    parse_fix_message,
    tokenize_fix_message,
)


class TestTokenizeFixMessage:
    """Tests for the tokenize_fix_message function."""

    def test_tokenize_pipe_delimiter(self) -> None:
        """Test tokenizing message with pipe delimiter."""
        msg = "8=FIX.4.4|35=8|55=AAPL"
        result = tokenize_fix_message(msg)
        assert result == {8: "FIX.4.4", 35: "8", 55: "AAPL"}

    def test_tokenize_soh_delimiter(self) -> None:
        """Test tokenizing message with SOH delimiter."""
        msg = "8=FIX.4.4\x0135=8\x0155=AAPL"
        result = tokenize_fix_message(msg)
        assert result == {8: "FIX.4.4", 35: "8", 55: "AAPL"}

    def test_tokenize_empty_message_raises_error(self) -> None:
        """Test that empty message raises FIXParseError."""
        with pytest.raises(FIXParseError):
            tokenize_fix_message("")

    def test_tokenize_whitespace_message_raises_error(self) -> None:
        """Test that whitespace-only message raises FIXParseError."""
        with pytest.raises(FIXParseError):
            tokenize_fix_message("   ")

    def test_tokenize_no_valid_pairs_raises_error(self) -> None:
        """Test that message with no valid pairs raises FIXParseError."""
        with pytest.raises(FIXParseError):
            tokenize_fix_message("invalid|message|format")

    def test_tokenize_skips_invalid_pairs(self) -> None:
        """Test that invalid pairs are skipped silently."""
        msg = "8=FIX.4.4|invalid|55=AAPL|bad=x"
        result = tokenize_fix_message(msg)
        # 'bad' is not a valid numeric tag, should be skipped
        # 'invalid' has no =, should be skipped
        assert result == {8: "FIX.4.4", 55: "AAPL"}

    def test_tokenize_handles_empty_value(self) -> None:
        """Test that empty values are preserved."""
        msg = "8=FIX.4.4|35=|55=AAPL"
        result = tokenize_fix_message(msg)
        assert result[35] == ""

    def test_tokenize_handles_value_with_equals(self) -> None:
        """Test that values containing = are preserved."""
        msg = "8=FIX.4.4|58=Reason=test|55=AAPL"
        result = tokenize_fix_message(msg)
        assert result[58] == "Reason=test"

    def test_tokenize_last_value_wins_for_duplicates(self) -> None:
        """Test that duplicate tags use the last value."""
        msg = "8=FIX.4.2|8=FIX.4.4|55=AAPL"
        result = tokenize_fix_message(msg)
        assert result[8] == "FIX.4.4"


class TestParseFixMessage:
    """Tests for the parse_fix_message function."""

    @pytest.fixture
    def valid_fix44_message(self) -> str:
        """Return a valid FIX 4.4 execution report message."""
        return (
            "8=FIX.4.4|35=8|37=ORD001|55=AAPL|54=1|32=100|31=150.50|"
            "30=NYSE|60=20240115-10:30:00.000|39=2|150=F|14=100|6=150.50"
        )

    @pytest.fixture
    def valid_fix42_message(self) -> str:
        """Return a valid FIX 4.2 execution report message."""
        return (
            "8=FIX.4.2|35=8|37=ORD002|55=GOOGL|54=2|32=50|31=142.75|"
            "30=NASDAQ|60=20240115-14:45:30.123|39=1"
        )

    @pytest.fixture
    def minimal_valid_message(self) -> str:
        """Return a minimal valid execution report message."""
        return "37=ORD003|55=MSFT|54=1|32=200|31=385.00|60=20240115-09:30:00"

    def test_parse_valid_fix44_message(self, valid_fix44_message: str) -> None:
        """Test parsing a valid FIX 4.4 message."""
        result = parse_fix_message(valid_fix44_message)

        assert isinstance(result, ExecutionReport)
        assert result.order_id == "ORD001"
        assert result.symbol == "AAPL"
        assert result.side == "BUY"
        assert result.quantity == 100.0
        assert result.price == 150.50
        assert result.venue == "NYSE"
        assert result.fill_type == "FULL"
        assert result.fix_version == "FIX.4.4"
        assert result.exec_type == "F"
        assert result.cum_qty == 100.0
        assert result.avg_px == 150.50

    def test_parse_valid_fix42_message(self, valid_fix42_message: str) -> None:
        """Test parsing a valid FIX 4.2 message."""
        result = parse_fix_message(valid_fix42_message)

        assert result.order_id == "ORD002"
        assert result.symbol == "GOOGL"
        assert result.side == "SELL"
        assert result.quantity == 50.0
        assert result.price == 142.75
        assert result.venue == "NASDAQ"
        assert result.fill_type == "PARTIAL"
        assert result.fix_version == "FIX.4.2"

    def test_parse_minimal_message(self, minimal_valid_message: str) -> None:
        """Test parsing a message with only required fields."""
        result = parse_fix_message(minimal_valid_message)

        assert result.order_id == "ORD003"
        assert result.symbol == "MSFT"
        assert result.side == "BUY"
        assert result.quantity == 200.0
        assert result.price == 385.00
        assert result.venue == "UNKNOWN"  # Default
        assert result.fill_type == "PARTIAL"  # Default
        assert result.fix_version == "FIX.4.4"  # Default
        assert result.exec_type is None
        assert result.cum_qty is None
        assert result.avg_px is None

    def test_parse_buy_side(self) -> None:
        """Test parsing buy order (side=1)."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=150|60=20240115-10:00:00"
        result = parse_fix_message(msg)
        assert result.side == "BUY"

    def test_parse_sell_side(self) -> None:
        """Test parsing sell order (side=2)."""
        msg = "37=ORD|55=AAPL|54=2|32=100|31=150|60=20240115-10:00:00"
        result = parse_fix_message(msg)
        assert result.side == "SELL"

    def test_parse_full_fill(self) -> None:
        """Test parsing full fill (ordStatus=2)."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=150|60=20240115-10:00:00|39=2"
        result = parse_fix_message(msg)
        assert result.fill_type == "FULL"

    def test_parse_partial_fill(self) -> None:
        """Test parsing partial fill (ordStatus=1)."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=150|60=20240115-10:00:00|39=1"
        result = parse_fix_message(msg)
        assert result.fill_type == "PARTIAL"

    def test_symbol_uppercase(self) -> None:
        """Test that symbol is uppercased."""
        msg = "37=ORD|55=aapl|54=1|32=100|31=150|60=20240115-10:00:00"
        result = parse_fix_message(msg)
        assert result.symbol == "AAPL"

    def test_parse_soh_delimiter(self) -> None:
        """Test parsing message with SOH delimiter."""
        msg = "37=ORD\x0155=AAPL\x0154=1\x0132=100\x0131=150\x0160=20240115-10:00:00"
        result = parse_fix_message(msg)
        assert result.symbol == "AAPL"


class TestParseFixMessageTimestamps:
    """Tests for timestamp parsing."""

    def test_timestamp_with_milliseconds(self) -> None:
        """Test parsing timestamp with milliseconds."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=150|60=20240115-10:30:45.123"
        result = parse_fix_message(msg)
        assert result.timestamp == datetime(2024, 1, 15, 10, 30, 45, 123000)

    def test_timestamp_without_milliseconds(self) -> None:
        """Test parsing timestamp without milliseconds."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=150|60=20240115-10:30:45"
        result = parse_fix_message(msg)
        assert result.timestamp == datetime(2024, 1, 15, 10, 30, 45)

    def test_timestamp_with_space_separator(self) -> None:
        """Test parsing timestamp with space separator."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=150|60=20240115 10:30:45.123"
        result = parse_fix_message(msg)
        assert result.timestamp == datetime(2024, 1, 15, 10, 30, 45, 123000)

    def test_invalid_timestamp_raises_error(self) -> None:
        """Test that invalid timestamp raises FIXValidationError."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=150|60=invalid-timestamp"
        with pytest.raises(FIXValidationError) as exc_info:
            parse_fix_message(msg)
        assert exc_info.value.field_tag == 60


class TestParseFixMessageErrors:
    """Tests for error handling in parse_fix_message."""

    def test_missing_order_id_raises_error(self) -> None:
        """Test that missing OrderID raises FIXMissingFieldError."""
        msg = "55=AAPL|54=1|32=100|31=150|60=20240115-10:00:00"
        with pytest.raises(FIXMissingFieldError) as exc_info:
            parse_fix_message(msg)
        assert exc_info.value.field_tag == 37
        assert "OrderID" in str(exc_info.value)

    def test_missing_symbol_raises_error(self) -> None:
        """Test that missing Symbol raises FIXMissingFieldError."""
        msg = "37=ORD|54=1|32=100|31=150|60=20240115-10:00:00"
        with pytest.raises(FIXMissingFieldError) as exc_info:
            parse_fix_message(msg)
        assert exc_info.value.field_tag == 55

    def test_missing_side_raises_error(self) -> None:
        """Test that missing Side raises FIXMissingFieldError."""
        msg = "37=ORD|55=AAPL|32=100|31=150|60=20240115-10:00:00"
        with pytest.raises(FIXMissingFieldError) as exc_info:
            parse_fix_message(msg)
        assert exc_info.value.field_tag == 54

    def test_missing_quantity_raises_error(self) -> None:
        """Test that missing LastQty raises FIXMissingFieldError."""
        msg = "37=ORD|55=AAPL|54=1|31=150|60=20240115-10:00:00"
        with pytest.raises(FIXMissingFieldError) as exc_info:
            parse_fix_message(msg)
        assert exc_info.value.field_tag == 32

    def test_missing_price_raises_error(self) -> None:
        """Test that missing LastPx raises FIXMissingFieldError."""
        msg = "37=ORD|55=AAPL|54=1|32=100|60=20240115-10:00:00"
        with pytest.raises(FIXMissingFieldError) as exc_info:
            parse_fix_message(msg)
        assert exc_info.value.field_tag == 31

    def test_missing_timestamp_raises_error(self) -> None:
        """Test that missing TransactTime raises FIXMissingFieldError."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=150"
        with pytest.raises(FIXMissingFieldError) as exc_info:
            parse_fix_message(msg)
        assert exc_info.value.field_tag == 60

    def test_invalid_side_raises_error(self) -> None:
        """Test that invalid Side raises FIXValidationError."""
        msg = "37=ORD|55=AAPL|54=3|32=100|31=150|60=20240115-10:00:00"
        with pytest.raises(FIXValidationError) as exc_info:
            parse_fix_message(msg)
        assert exc_info.value.field_tag == 54
        assert exc_info.value.field_value == "3"

    def test_invalid_quantity_raises_error(self) -> None:
        """Test that non-numeric quantity raises FIXValidationError."""
        msg = "37=ORD|55=AAPL|54=1|32=abc|31=150|60=20240115-10:00:00"
        with pytest.raises(FIXValidationError) as exc_info:
            parse_fix_message(msg)
        assert exc_info.value.field_tag == 32

    def test_invalid_price_raises_error(self) -> None:
        """Test that non-numeric price raises FIXValidationError."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=xyz|60=20240115-10:00:00"
        with pytest.raises(FIXValidationError) as exc_info:
            parse_fix_message(msg)
        assert exc_info.value.field_tag == 31

    def test_empty_message_raises_error(self) -> None:
        """Test that empty message raises FIXParseError."""
        with pytest.raises(FIXParseError):
            parse_fix_message("")


class TestExecutionReportModel:
    """Tests for the ExecutionReport Pydantic model."""

    def test_valid_model_creation(self) -> None:
        """Test creating a valid ExecutionReport."""
        report = ExecutionReport(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            quantity=100.0,
            price=150.50,
            venue="NYSE",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            fill_type="FULL",
        )
        assert report.order_id == "ORD001"
        assert report.symbol == "AAPL"

    def test_negative_quantity_raises_error(self) -> None:
        """Test that negative quantity raises ValidationError."""
        with pytest.raises(ValueError):
            ExecutionReport(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                quantity=-100.0,
                price=150.50,
                venue="NYSE",
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                fill_type="FULL",
            )

    def test_negative_price_raises_error(self) -> None:
        """Test that negative price raises ValidationError."""
        with pytest.raises(ValueError):
            ExecutionReport(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                quantity=100.0,
                price=-150.50,
                venue="NYSE",
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                fill_type="FULL",
            )

    def test_empty_order_id_raises_error(self) -> None:
        """Test that empty order_id raises ValidationError."""
        with pytest.raises(ValueError):
            ExecutionReport(
                order_id="",
                symbol="AAPL",
                side="BUY",
                quantity=100.0,
                price=150.50,
                venue="NYSE",
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                fill_type="FULL",
            )

    def test_invalid_side_raises_error(self) -> None:
        """Test that invalid side raises ValidationError."""
        with pytest.raises(ValueError):
            ExecutionReport(
                order_id="ORD001",
                symbol="AAPL",
                side="INVALID",  # type: ignore[arg-type]
                quantity=100.0,
                price=150.50,
                venue="NYSE",
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                fill_type="FULL",
            )


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_very_large_order(self) -> None:
        """Test parsing a very large order quantity."""
        msg = "37=ORD|55=AAPL|54=1|32=1000000000|31=150|60=20240115-10:00:00"
        result = parse_fix_message(msg)
        assert result.quantity == 1_000_000_000.0

    def test_fractional_quantity(self) -> None:
        """Test parsing fractional quantity."""
        msg = "37=ORD|55=AAPL|54=1|32=100.5|31=150|60=20240115-10:00:00"
        result = parse_fix_message(msg)
        assert result.quantity == 100.5

    def test_high_precision_price(self) -> None:
        """Test parsing high precision price."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=150.123456|60=20240115-10:00:00"
        result = parse_fix_message(msg)
        assert result.price == 150.123456

    def test_long_symbol(self) -> None:
        """Test parsing a longer symbol."""
        msg = "37=ORD|55=BRK.A|54=1|32=100|31=150|60=20240115-10:00:00"
        result = parse_fix_message(msg)
        assert result.symbol == "BRK.A"

    def test_special_characters_in_order_id(self) -> None:
        """Test parsing order ID with special characters."""
        msg = "37=ORD-001_ABC|55=AAPL|54=1|32=100|31=150|60=20240115-10:00:00"
        result = parse_fix_message(msg)
        assert result.order_id == "ORD-001_ABC"

    def test_unknown_tags_ignored(self) -> None:
        """Test that unknown tags are ignored."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=150|60=20240115-10:00:00|9999=unknown"
        result = parse_fix_message(msg)
        assert result.order_id == "ORD"  # Parsing still works

    def test_midnight_timestamp(self) -> None:
        """Test parsing midnight timestamp."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=150|60=20240115-00:00:00.000"
        result = parse_fix_message(msg)
        assert result.timestamp.hour == 0
        assert result.timestamp.minute == 0

    def test_end_of_day_timestamp(self) -> None:
        """Test parsing end of day timestamp."""
        msg = "37=ORD|55=AAPL|54=1|32=100|31=150|60=20240115-23:59:59.999"
        result = parse_fix_message(msg)
        assert result.timestamp.hour == 23
        assert result.timestamp.minute == 59
        assert result.timestamp.second == 59
