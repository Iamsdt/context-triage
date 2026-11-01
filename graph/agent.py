from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from injectq import InjectQ
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import SqliteCheckpointer

from graph.state import UnforgetDemoState
from graph.tools import list_project_files, read_service_log
from unforget import UnforgetContextManager, recall


load_dotenv()

# Threads live in one local SQLite file next to this project, so they survive restarts of
# `10xgraph dev`. Delete unforget.db to start over.
checkpointer = SqliteCheckpointer(Path(__file__).resolve().parent.parent / "unforget.db")

# Dependency injection container; 10xgraph.json points at it ("injectq": "graph.agent:container").
container = InjectQ.get_instance()

# Demo values: a small budget so decisions show up after a few turns in the playground.
# Credentials (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, ANTHROPIC_API_KEY) are read
# when a trim runs, not at import. Without the Cloudflare pair the context is left as is.
context_manager = UnforgetContextManager(
    token_budget=3000,  # trim when the context estimate passes this
    target_ratio=0.6,  # aim for about 1800 tokens after a trim
    keep_recent_turns=4,  # the last 4 turns are never touched
    min_confidence=0.7,  # apply Clef's top option only at p >= 0.7, otherwise keep
    clef_model="clef-flash",  # or "clef" for the larger model
    writer_model="claude-haiku-5-5",  # writes compress notes and pinned facts
    background=True,  # Clef and the writer run off the request path; applied on the next call
    hard_budget_ratio=2.0,  # past 6000 tokens a call waits for the job (or trims inline)
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
                Earlier parts of this conversation may appear as a memory summary with facts,
                notes and archived items. When an archived item holds a detail you need, call
                recall with its number instead of guessing.
                Follow the user's stated preferences and decisions.
            """,
        },
        {
            "role": "user",
            "content": f"Today Date is {datetime.now().strftime('%Y-%m-%d')}",
        },
    ],
    trim_context=True,
    tools=[read_service_log, list_project_files, recall],
    context_manager=context_manager,
    container=container,
)

app = react_agent.compile(checkpointer=checkpointer)
