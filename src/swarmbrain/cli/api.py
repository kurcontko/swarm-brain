from __future__ import annotations

import uvicorn

from swarmbrain.application.runtime import build_runtime
from swarmbrain.config import ApiSettings
from swarmbrain.transports.http import create_app


def main() -> None:
    settings = ApiSettings.from_env()
    app = create_app(
        build_runtime(settings),
        console_demo=settings.console_demo_enabled,
        public_docs=settings.public_docs_enabled,
    )
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
