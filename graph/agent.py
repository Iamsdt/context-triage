from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from injectq import InjectQ
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import SqliteCheckpointer

from graph.state import UnforgetDemoState
from graph.tools import list_project_files, read_service_log
from unforget import UnforgetContextManager


load_dotenv()

# Threads live in one local SQLite file next to this project, so they survive restarts of
# `10xgraph dev`. Delete unforget.db to start over.
checkpointer = SqliteCheckpointer(Path(__file__).resolve().parent.parent / "unforget.db")

# Dependency injection container; 10xgraph.json points at it ("injectq": "graph.agent:container").
container = InjectQ.get_instance()

# Smoke-test values: a tiny budget and one protected turn so even a short chat triggers a check.
# For the real demo use about 6000 tokens and 4 recent turns.
# Credentials (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, ANTHROPIC_API_KEY) are read
# when a trim runs, not at import. Without the Cloudflare pair the context is left as is.
context_manager = UnforgetContextManager(
    # tiny budget for a first smoke test so even "hi" / "hello" triggers a check
    # (raise it to about 6000 for the real demo)
    token_budget=500,
    keep_recent_turns=1,  # only the latest turn is protected (smoke test; use 4 for the real demo)
    min_confidence=0.7,  # apply Clef's top option only at p >= 0.7, otherwise keep
    clef_model="clef",  # the REST endpoint accepts only "clef" (clef-flash gave HTTP 422)
    writer_model="claude-haiku-5-5",  # writes the compress notes
)

react_agent = ReactAgent(
    state=UnforgetDemoState(),
    model="claude-sonnet-5-5",
    system_prompt=[
        {
            "role": "system",
            "content": """
                You are a helpful engineering assistant for a demo online shop called shopfront.
                Use read_service_log to look at service logs and list_project_files to look at
                the repository. Both return demo data.
                Earlier parts of this conversation may appear as short notes.
                Follow the user's stated preferences and decisions.
            """,
        },
        {
            "role": "user",
            "content": f"Today Date is {datetime.now().strftime('%Y-%m-%d')}",
        },
    ],
    trim_context=False,  # core trims once at the end of each run anyway
    tools=[read_service_log, list_project_files],
    context_manager=context_manager,
    container=container,
)

app = react_agent.compile(checkpointer=checkpointer)
