"""Public import surface and packaged-resource smoke tests."""

from importlib.resources import files

import CellChatPy


def test_all_declared_public_names_are_importable() -> None:
    missing = [name for name in CellChatPy.__all__ if not hasattr(CellChatPy, name)]
    assert missing == []


def test_package_version_matches_release_metadata() -> None:
    assert CellChatPy.__version__ == "1.1.0"


def test_bundled_database_resources_are_present() -> None:
    package_root = files("CellChatPy")
    for species in ("human", "mouse", "zebrafish"):
        database_dir = package_root.joinpath("CellChatDB", f"CellChatDB.{species}")
        for filename in ("interaction.csv", "complex.csv", "cofactor.csv", "geneInfo.csv"):
            assert database_dir.joinpath(filename).is_file(), f"missing {species}/{filename}"
