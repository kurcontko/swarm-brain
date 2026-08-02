from __future__ import annotations

import uvicorn

from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.config import ApiSettings
from swarmbrain.transports.http import create_app


def main() -> None:
    settings = ApiSettings.from_env()
    if settings.database_url:
        raise RuntimeError(
            "the v0 executable currently supports the in-memory adapter; "
            "unset SWARMBRAIN_DATABASE_URL or compose a Cockroach store explicitly"
        )
    app = create_app(build_in_memory_runtime(settings.token_secret))
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
