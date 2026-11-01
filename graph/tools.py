"""Demo tools that return long-ish output, so turns carry tool results worth archiving.

Both tools read from a small built-in demo dataset: the services, files and log lines are
generated, not real. The output is deterministic per input, so a recalled turn can be
checked against a fresh call.
"""

import zlib
from datetime import UTC, datetime, timedelta


_SERVICES = ("checkout-api", "payments", "inventory", "search", "auth")
_MIN_LINES = 10
_MAX_LINES = 80

_INFO = (
    "GET /health 200 2ms",
    "request completed status=200 route=/v1/cart duration_ms={ms}",
    "cache hit ratio={ratio}% over last 60s",
    "worker pool size=8 busy={busy}",
    "request completed status=200 route=/v1/orders duration_ms={ms}",
    "connection pool to postgres-primary: active={busy} idle=12",
)
_WARN = (
    "slow query on orders_by_customer took {ms}ms (threshold 250ms)",
    "retrying call to payments (attempt 2/3) after timeout",
    "queue depth for email-outbox at {busy}00, above soft limit",
)
_ERROR = (
    "upstream timeout calling payments after 3000ms order_id=A-{order}",
    "card authorization failed: gateway returned 502 order_id=A-{order}",
    "deadlock detected on table inventory_reservations, transaction rolled back",
)

_FILES = (
    ("README.md", 4_812),
    ("pyproject.toml", 1_903),
    ("Dockerfile", 744),
    ("docker-compose.yml", 1_310),
    ("src/shopfront/__init__.py", 112),
    ("src/shopfront/app.py", 6_208),
    ("src/shopfront/config.py", 2_455),
    ("src/shopfront/routes/cart.py", 9_871),
    ("src/shopfront/routes/checkout.py", 14_302),
    ("src/shopfront/routes/orders.py", 8_066),
    ("src/shopfront/services/payments.py", 11_540),
    ("src/shopfront/services/inventory.py", 7_219),
    ("src/shopfront/services/email.py", 3_377),
    ("src/shopfront/models/order.py", 5_104),
    ("src/shopfront/models/customer.py", 3_980),
    ("src/shopfront/db/migrations/0001_init.sql", 6_651),
    ("src/shopfront/db/migrations/0002_reservations.sql", 2_048),
    ("tests/test_cart.py", 5_530),
    ("tests/test_checkout.py", 9_912),
    ("tests/test_payments.py", 7_403),
    ("scripts/seed_demo_data.py", 2_771),
    (".github/workflows/ci.yml", 1_655),
)


def _rng(seed: str):
    """Small deterministic number stream from a text seed (no global random state)."""
    state = zlib.crc32(seed.encode("utf-8")) or 1
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state


def read_service_log(service: str, lines: int = 40) -> str:
    """Read the most recent log lines of a service in the demo shop (demo data).

    Available services: checkout-api, payments, inventory, search, auth.

    Args:
        service: Service name, for example "checkout-api".
        lines: How many recent lines to return (10 to 80).
    """
    name = service.strip().lower()
    if name not in _SERVICES:
        return f"Unknown service {service!r}. Available: {', '.join(_SERVICES)}."
    count = max(_MIN_LINES, min(_MAX_LINES, int(lines)))
    numbers = _rng(name)
    start = datetime(2026, 10, 31, 14, 0, 0, tzinfo=UTC)
    out = [f"# {name}: last {count} lines (demo data)"]
    for i in range(count):
        n = next(numbers)
        when = start + timedelta(seconds=i * 7 + n % 5)
        roll = n % 20
        if roll == 0:
            level, template = "ERROR", _ERROR[n % len(_ERROR)]
        elif roll < 3:  # noqa: PLR2004
            level, template = "WARN", _WARN[n % len(_WARN)]
        else:
            level, template = "INFO", _INFO[n % len(_INFO)]
        text = template.format(
            ms=40 + n % 900, ratio=70 + n % 29, busy=1 + n % 8, order=10_000 + n % 90_000
        )
        out.append(f"{when:%Y-%m-%dT%H:%M:%SZ} {level:<5} {name} {text}")
    return "\n".join(out)


def list_project_files(prefix: str = "") -> str:
    """List files in the demo "shopfront" repository with their sizes (demo data).

    Args:
        prefix: Only list paths that start with this, for example "src/shopfront/routes".
            Empty lists everything.
    """
    wanted = prefix.strip()
    wanted = wanted.removeprefix("./")
    rows = [(path, size) for path, size in _FILES if path.startswith(wanted)]
    if not rows:
        return f"No files under {prefix!r} in the demo repository."
    total = sum(size for _, size in rows)
    body = "\n".join(f"{size:>8,d}  {path}" for path, size in rows)
    return f"shopfront/ ({len(rows)} files, {total:,d} bytes, demo data)\n{body}"
