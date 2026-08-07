"""The console page is public, inert, and never a way around authentication."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from conftest import make_actor
from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.transports.http import create_app
from swarmbrain.transports.http.console import console_html

SECRET = "console-test-secret-0123456789ab"


def _console_script() -> str:
    """The page's inline script, which is all of its behaviour."""
    body = console_html()
    return body[body.index("<script>") + len("<script>") : body.rindex("</script>")]


def _markup_helpers() -> str:
    """The escaping block, lifted verbatim so the test exercises shipped code."""
    script = _console_script()
    return script[script.index("  const esc = (value) =>") : script.index("  // -- helpers")]


@pytest.mark.asyncio
async def test_openapi_surface_is_absent_unless_explicitly_enabled() -> None:
    closed = create_app(build_in_memory_runtime(SECRET))
    opened = create_app(build_in_memory_runtime(SECRET), public_docs=True)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=closed), base_url="http://docs-test"
    ) as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert (await client.get(path)).status_code == 404
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=opened), base_url="http://docs-test"
    ) as client:
        assert (await client.get("/openapi.json")).status_code == 200


@pytest.mark.asyncio
async def test_console_page_is_served_unauthenticated_and_self_contained() -> None:
    app = create_app(build_in_memory_runtime(SECRET))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://console-test") as client:
        response = await client.get("/console")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert body.startswith("<!doctype html>")
    assert "swarmbrain" in body
    # Offline-capable: no external origin is referenced by the document.
    assert "http://" not in body
    assert "https://" not in body
    assert "//cdn" not in body
    # The page holds no secret, no token, and no run data.
    assert SECRET not in body
    assert "Bearer ey" not in body
    assert "sessionStorage" in body


@pytest.mark.asyncio
async def test_console_does_not_relax_data_route_authentication(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(scope_ids)
    token = runtime.tokens.issue(actor)
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    run_id = scope_ids["run_id"]

    async with httpx.AsyncClient(transport=transport, base_url="http://console-test") as client:
        for path in (
            f"/v1/runs/{run_id}/events",
            f"/v1/runs/{run_id}/metrics",
            "/v1/memories/00000000-0000-0000-0000-000000000000/lineage",
        ):
            anonymous = await client.get(path)
            assert anonymous.status_code == 401
            assert anonymous.json()["error"]["code"] == "authentication_required"
            assert anonymous.headers["WWW-Authenticate"] == "Bearer"

        headers = {"Authorization": f"Bearer {token}"}
        events = await client.get(f"/v1/runs/{run_id}/events", headers=headers)
        assert events.status_code == 200
        assert events.json()["events"] == []
        metrics = await client.get(f"/v1/runs/{run_id}/metrics", headers=headers)
        assert metrics.status_code == 200
        assert metrics.json()["run_id"] == run_id

        # The console route stays a page, never a data endpoint.
        other_run = await client.get(
            "/v1/runs/00000000-0000-0000-0000-000000000000/events", headers=headers
        )
        assert other_run.status_code == 404


def test_console_document_reads_only_the_canonical_routes() -> None:
    body = console_html()

    assert "/v1/runs/${encodeURIComponent(state.runId)}/events" in body
    assert "/v1/runs/${encodeURIComponent(state.runId)}/metrics" in body
    assert "/v1/memories/${encodeURIComponent(memoryId)}/lineage" in body
    assert "events:read" in body and "metrics:read" in body and "memory:recall" in body

    # Read-only against the data plane: the only POST the page can issue is the
    # operator-gated demo trigger, and it never carries the viewer's bearer token.
    api_call = body.index("async function api(path)")
    assert 'method: "POST"' not in body[api_call : body.index("async function pullEvents")]
    posts = body.count('method: "POST"')
    assert posts == 1
    assert "`${state.base}/console/demo`" in body
    assert "/v1/" not in body[body.index("async function triggerDemo") :]


def test_console_prefers_enriched_payload_labels_with_uuid_fallback() -> None:
    body = console_html()

    # The board, the roster, and the ledger all read the additive payload keys.
    assert "task_title" in body
    assert "agent_display" in body
    assert "agent_harness" in body
    # ...and every label degrades to the short id when an event predates them.
    assert "state.agentLabels.get(agentId)) || short(agentId)" in body
    assert "short(task.taskId)" in body


def test_console_has_a_single_markup_sink_fed_only_by_the_escaping_tag() -> None:
    """No page-rendered string can become markup except through html``.

    Every value the console draws is attacker-influenceable: an agent holding
    memory:publish writes the titles, contents and payload values a public
    viewer's browser later renders, and error text arrives from a remote API.
    So the page is only safe if there is exactly one place markup reaches the
    parser and it cannot be handed a raw string. That is what this asserts.
    """
    script = _console_script()

    writes = [line.strip() for line in script.splitlines() if ".innerHTML" in line]
    assert writes == [
        "if (isMarkup(value)) node.innerHTML = value.text;",
        "else if (Array.isArray(value) && value.every(isMarkup)) "
        "node.innerHTML = interpolate(value);",
    ]
    # Both writes live in setHTML, and every value they take was built by html``.
    assert script.count("innerHTML") == 2
    assert script.index("const setHTML") < script.index("node.innerHTML")
    assert script.index("node.innerHTML = interpolate(value)") < script.index("// -- helpers")

    # ...and there is no second route to the HTML parser.
    for sink in (
        "insertAdjacentHTML",
        "outerHTML",
        "document.write",
        "srcdoc",
        "eval(",
        "new Function",
        "javascript:",
        "createContextualFragment",
        "setHTMLUnsafe",
    ):
        assert sink not in script, sink

    # Escaping is never applied by hand, so a render function cannot forget it
    # and no value can be escaped twice: the tag is the only caller.
    assert script.count("esc(") == 1
    assert "return esc(value);" in script

    # The renderers do go through the sink rather than assigning markup directly.
    assert script.count("setHTML(") >= 10
    assert script.count("html`") >= 40


HELPER_ASSERTIONS = r"""
const fail = (message) => { console.error(message); process.exit(1); };
const eq = (got, want, label) => {
  if (got !== want) fail(`${label}\n  got  ${got}\n  want ${want}`);
};

// Every character that can end a text node or a quoted attribute is encoded,
// and the ampersand is rewritten first so no escape can be re-opened.
eq(esc("<script>alert(1)</script>"),
   "&lt;script&gt;alert(1)&lt;/script&gt;", "text context");
eq(esc('" onerror="alert(1)'), "&quot; onerror=&quot;alert(1)", "double-quoted attribute");
eq(esc("' onerror='alert(1)"), "&#39; onerror=&#39;alert(1)", "single-quoted attribute");
eq(esc("&lt;"), "&amp;lt;", "ampersand is escaped, not passed through");
eq(esc("</style><svg onload=alert(1)>"),
   "&lt;/style&gt;&lt;svg onload=alert(1)&gt;", "style break-out");

// Absent values render as nothing rather than as "null"/"undefined", and a
// zero counter is not mistaken for absence.
eq(esc(null), "", "null");
eq(esc(undefined), "", "undefined");
eq(esc(0), "0", "zero survives");
eq(esc(false), "false", "false survives");

// The tag escapes interpolations in both contexts while keeping its own markup.
const payload = "<img src=x onerror=alert(1)>";
eq(html`<p title="${payload}">${payload}</p>`.text,
   '<p title="&lt;img src=x onerror=alert(1)&gt;">&lt;img src=x onerror=alert(1)&gt;</p>',
   "tagged template");

// Composition never double-escapes, and arrays concatenate without a separator.
const inner = html`<b>${"a&b"}</b>`;
eq(html`${inner}`.text, "<b>a&amp;b</b>", "nested markup is not re-escaped");
eq(html`${[inner, inner]}`.text, "<b>a&amp;b</b><b>a&amp;b</b>", "array of markup");
eq(html`${["a&b", "<i>"]}`.text, "a&amp;b&lt;i&gt;", "array of plain values");

// A bare string is never markup, so a render function that forgets the tag
// produces a visible escape instead of an injection.
const sink = () => ({ innerHTML: "untouched", textContent: "untouched" });
let node = sink();
setHTML(node, "<img src=x onerror=alert(1)>");
eq(node.innerHTML, "untouched", "a bare string does not reach innerHTML");
eq(node.textContent, "<img src=x onerror=alert(1)>", "...it lands as text");

// ...and neither is an object shaped to look like markup, because the brand is
// a private symbol that untrusted JSON cannot carry.
node = sink();
setHTML(node, JSON.parse('{"text": "<img src=x onerror=alert(1)>"}'));
eq(node.innerHTML, "untouched", "a forged markup object is not trusted");

node = sink();
setHTML(node, html`<em>ok</em>`);
eq(node.innerHTML, "<em>ok</em>", "html`` markup does reach innerHTML");

node = sink();
setHTML(node, [html`<em>a</em>`, "<em>b</em>"]);
eq(node.innerHTML, "untouched", "a mixed array is not trusted");
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_console_escape_helper_neutralises_injection_in_every_context(
    tmp_path: Path,
) -> None:
    """Run the page's own escaping block and try to break out of it."""
    harness = tmp_path / "markup.js"
    harness.write_text(_markup_helpers() + HELPER_ASSERTIONS, encoding="utf-8")

    result = subprocess.run(  # noqa: S603
        [shutil.which("node") or "node", str(harness)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
