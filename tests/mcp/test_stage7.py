import pytest
from pydantic import ValidationError

from tgarchive.mcp.server import _SearchFiltersWire


def test_search_filters_wire_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        _SearchFiltersWire.model_validate({"chat_id": [1]})
