"""Initial tests to verify project setup."""

import src


def test_version() -> None:
    """Verify package version is set."""
    assert src.__version__ == "0.1.0"


def test_import_parsers() -> None:
    """Verify parsers module is importable."""
    from src import parsers

    assert parsers is not None


def test_import_agents() -> None:
    """Verify agents module is importable."""
    from src import agents

    assert agents is not None


def test_import_evaluation() -> None:
    """Verify evaluation module is importable."""
    from src import evaluation

    assert evaluation is not None


def test_import_observability() -> None:
    """Verify observability module is importable."""
    from src import observability

    assert observability is not None
