"""Framework-neutral cancellation signal shared by services and jobs.

Kept dependency-free (no Flask/api imports) so services under
``*_services.py`` can raise it directly without importing anything from
``api/``; ``api/jobs.py`` catches it to move a ``Job`` to the ``CANCELLED``
terminal status instead of ``FAILED``.
"""

from __future__ import annotations


class OperationCancelled(Exception):
    """Raised to unwind a long-running operation whose job was cancelled."""
