"""Ordered download sources for the pinned Microsoft ONNX Runtime artifacts.

Every artifact is tried on the Kapsl R2 mirror first and on its upstream GitHub URL second.
The pinned digests live in the fetchers, never here and never on the mirror: each source's
bytes go through the same checks, and a source that fails them is skipped, so a wrong file
is never accepted from either place.

KAPSL_ORT_MIRROR_BASE_URL overrides the mirror base (an empty value disables the mirror).
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from typing import Callable, Sequence, Tuple, TypeVar

MIRROR_ENV = "KAPSL_ORT_MIRROR_BASE_URL"
DEFAULT_MIRROR_BASE_URL = "https://downloads.kapsl.net/mirror/onnxruntime"
USER_AGENT = "kapsl-integrations-ort-packager/0.1"

T = TypeVar("T")
Source = Tuple[str, Callable[[bytes], T]]


def mirror_url(version: str, relative_path: str) -> str | None:
    """Mirror URL of <relative_path> under the version directory, or None when disabled."""
    base = os.environ.get(MIRROR_ENV, DEFAULT_MIRROR_BASE_URL).strip().rstrip("/")
    if not base:
        return None
    return f"{base}/v{version}/{relative_path}"


def fetch_first_valid(
    sources: Sequence[Source[T]],
    error_type: type[Exception],
    max_bytes: int,
    timeout: float,
    label: str,
) -> T:
    """Return accept(payload) for the first source whose payload passes accept.

    accept raises error_type to reject a payload. Network and rejection errors move on to
    the next source; when none succeeds every failure is reported in one error.
    """
    failures: list[str] = []
    for url, accept in sources:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read(max_bytes + 1)
            return accept(payload)
        except (urllib.error.URLError, TimeoutError, OSError, error_type) as error:
            failures.append(f"{url}: {error}")
            print(
                f"warning: {label}: skipping {url}: {error}",
                file=sys.stderr,
                flush=True,
            )
    raise error_type(f"no source provided a verified {label}: " + "; ".join(failures))
