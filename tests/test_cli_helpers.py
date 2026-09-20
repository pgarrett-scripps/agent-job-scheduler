"""Parsing helpers. User-facing and easy to get subtly wrong, so pinned down."""

import pytest

from ajs.cli import detect_project, parse_duration, parse_mem


class TestParseDuration:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("30s", 30),
            ("10m", 600),
            ("2h", 7200),
            ("1d", 86400),
            ("90", 90),
            ("1.5h", 5400),
            ("0.5m", 30),
            ("  10m  ", 600),
            ("10M", 600),
        ],
    )
    def test_accepted_forms(self, text, seconds):
        assert parse_duration(text) == seconds

    @pytest.mark.parametrize("text", ["", "   ", "abc", "10x"])
    def test_rejected_forms(self, text):
        with pytest.raises(ValueError):
            parse_duration(text)


class TestParseMem:
    @pytest.mark.parametrize(
        ("text", "mb"),
        [("512M", 512), ("8G", 8192), ("512", 512), ("1.5G", 1536), ("2g", 2048), (" 4G ", 4096)],
    )
    def test_accepted_forms(self, text, mb):
        assert parse_mem(text) == mb

    @pytest.mark.parametrize("text", ["", "abc", "8T"])
    def test_rejected_forms(self, text):
        with pytest.raises(ValueError):
            parse_mem(text)


class TestDetectProject:
    def test_uses_the_git_repo_name(self, tmp_path):
        import subprocess

        repo = tmp_path / "my-repo"
        (repo / "sub" / "deep").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        # Submitting from a subdirectory must still attribute to the repo, or fair-share
        # would treat each subdirectory as a separate project.
        assert detect_project(repo / "sub" / "deep") == "my-repo"

    def test_falls_back_to_the_directory_name(self, tmp_path):
        plain = tmp_path / "not-a-repo"
        plain.mkdir()
        assert detect_project(plain) == "not-a-repo"
