#!/usr/bin/env python
"""Example: Using remote MCP server with CodeAct-style function generation.

This example demonstrates how to connect to a remote MCP server (BioContext.AI)
and use its biomedical knowledge base tools through generated Python wrapper functions.

BioContext.AI provides access to:
- UniProt protein information
- AlphaFold structure predictions
- Human protein-protein interaction networks
- Gene-disease associations
- Drug-target interactions
- And more...

The agent automatically generates a Python module (biocontext_mcp.py) with typed
wrapper functions that can be imported and used in scripts.
"""

from pathlib import Path

from dotenv import load_dotenv

from alfredo import Agent
from alfredo.integrations.langchain import create_langchain_tools

# Load environment variables
load_dotenv()

# Create workspace directory
workspace = Path(__file__).parent.parent / "notebooks" / "sandbox"
workspace.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run the remote MCP server example."""
    print("=" * 70)
    print("Remote MCP Server Example: BioContext.AI")
    print("=" * 70)
    print()

    # Configure remote MCP server
    # Note: Alfredo automatically handles SSE responses and sessionless authentication
    codeact_mcp_functions = {
        "biocontext": {
            "url": "https://mcp.biocontext.ai/mcp/",  # Remote MCP server
        }
    }

    # Get standard Alfredo tools
    alfredo_tools = create_langchain_tools(cwd=str(workspace))

    # Create agent with remote MCP server
    print("Creating agent with BioContext.AI MCP server...")
    agent = Agent(
        cwd=str(workspace),
        model_name="gpt-4.1-mini",
        codeact_mcp_functions=codeact_mcp_functions,
        tools=alfredo_tools,
        verbose=True,
    )

    print()
    print("=" * 70)
    print("Agent created successfully!")
    print("=" * 70)
    print(f"Working directory: {workspace}")
    print(f"Generated module: {workspace / 'biocontext_mcp.py'}")
    print()

    # Run a biomedical task
    print("=" * 70)
    print("Running biomedical task...")
    print("=" * 70)
    print()

    agent.run("""
Write a Python script that leverages available biocontext functions to:

1. Get the interactors of the TP53 gene (use gene symbol 'TP53')
2. Save the list of interactors to a file called 'tp53_interactors.txt'
3. Print a summary of how many interactors were found

The script should use the biocontext_mcp module functions.
""")

    # Display execution trace
    print()
    print("=" * 70)
    print("Execution Trace")
    print("=" * 70)
    agent.display_trace()

    # Show results
    print()
    print("=" * 70)
    print("Final Answer")
    print("=" * 70)
    print(agent.results.get("final_answer", "No answer provided"))

    # Check if output file was created
    output_file = workspace / "tp53_interactors.txt"
    if output_file.exists():
        print()
        print("=" * 70)
        print("Output File Created")
        print("=" * 70)
        print(f"File: {output_file}")
        print(f"Size: {output_file.stat().st_size} bytes")
        print()
        print("First 500 characters:")
        print(output_file.read_text()[:500])


if __name__ == "__main__":
    main()
