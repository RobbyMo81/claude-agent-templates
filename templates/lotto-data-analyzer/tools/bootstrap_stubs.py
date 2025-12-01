# tools/bootstrap_stubs.py
"""
Drop minimal render_page() stubs into each core module so Streamlit can load
the app long before you finish writing every analysis page.
"""
import textwrap, pathlib

CORE_FILES = [
    "ingest", "csv_formatter", "frequency", "dow_analysis",
    "time_trends", "inter_draw", "combos", "sums", "ml_experimental",
    # "llm_query",   # Uncomment when you're ready
]

TEMPLATE = textwrap.dedent("""\
    import streamlit as st

    def render_page():
        st.header("{title}")
        st.info("🚧  Work in progress  –  check back soon!")
""")

root = pathlib.Path(__file__).resolve().parents[1] / "core"
root.mkdir(exist_ok=True)

for name in CORE_FILES:
    path = root / f"{name}.py"
    if path.exists():
        continue                      # keep any real code you've started
    path.write_text(TEMPLATE.format(title=name.replace('_', ' ').title()))
    print(f"✔  {path.relative_to(root.parent)}")
