from pathlib import Path
import pytest

ALLOWED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a"}


def is_allowed_file(filename: str | None) -> bool:
    """許可されたファイル形式かチェック"""
    if filename is not None:
        return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS
    return False

@pytest.mark.parametrize(('str', 'expected'),
    [
        ('aaa', False),
        ('aaa.mp4', True),
        ('', False),
        (None, False)
    ]
)

def test_is_allowed_file(str, expected):
    assert is_allowed_file(str) == expected