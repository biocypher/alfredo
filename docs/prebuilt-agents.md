# Prebuilt Agents

Alfredo includes specialized prebuilt agents for common tasks. These agents wrap the base `Agent` class with custom prompts and configurations optimized for specific use cases.

## Available Agents

- **[ExplorationAgent](#explorationagent)** - Directory exploration and markdown report generation
- **[ReflexionAgent](#reflexionagent)** - Research with iterative self-critique and web search

---

## ExplorationAgent

**Purpose**: Explore directories and generate comprehensive markdown reports with smart file reading and data analysis.

The ExplorationAgent autonomously navigates a directory structure, categorizes files, analyzes data files with pandas, and produces a well-structured markdown report.

### Features

- **Smart File Size Handling** - Automatically adjusts reading strategy based on file size
- **Data File Analysis** - Generates pandas analysis for CSV, Excel, Parquet, HDF5, JSON files
- **File Categorization** - Groups files by type (code, data, config, docs, binary)
- **Context Steering** - Use prompts to focus exploration on specific aspects
- **Configurable Thresholds** - Customize size limits and preview line counts

### Basic Usage

```python
from alfredo.prebuilt import ExplorationAgent

# Basic exploration
agent = ExplorationAgent(
    cwd="./my_project",
    output_path="./reports/project_overview.md"
)

report = agent.explore()
print(report)
```

### Context Steering

Guide the exploration with a context prompt:

```python
# Focus on specific aspects
agent = ExplorationAgent(
    cwd="./data_pipeline",
    context_prompt="""
Focus on data schemas, transformations, and quality checks.
Note any data validation logic and preprocessing steps.
Identify potential data quality issues.
    """,
    model_name="gpt-4.1-mini"
)

report = agent.explore()
```

### Advanced Configuration

```python
agent = ExplorationAgent(
    cwd="./large_project",
    context_prompt="Focus on API endpoints and authentication mechanisms",
    max_file_size_bytes=50_000,  # More aggressive size limit (50KB)
    preview_kb=25,               # Preview first 25KB for large files
    output_path="./reports/api_exploration.md",
    model_name="gpt-4o-mini",
    verbose=True
)

report = agent.explore()

# View execution trace
agent.display_trace()
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cwd` | `"."` | Directory to explore |
| `context_prompt` | `None` | Optional context to steer exploration focus |
| `model_name` | `"gpt-4.1-mini"` | Model to use for exploration |
| `max_file_size_bytes` | `100_000` | Threshold for limited reading (100KB) |
| `preview_kb` | `50` | Size in KB to preview for large files |
| `preview_lines` | `None` | Number of lines to preview (alternative to preview_kb) |
| `output_path` | `None` | Report save path (default: `notes/exploration_report.md`) |
| `verbose` | `True` | Print progress updates |

### How It Works

**1. File Discovery**

Lists all files recursively to understand directory structure:

```python
# Agent uses list_files(recursive=true)
# Output: Full directory tree with file sizes
```

**2. File Categorization**

Groups files by type:
- **Source Code**: .py, .r, .js, .ts, .java, .go, etc.
- **Data Files**: .csv, .xlsx, .parquet, .h5, .json, etc.
- **Configuration**: .yaml, .json, .toml, .env, .ini
- **Documentation**: .md, .txt, .rst
- **Binary/Other**: Images, archives, executables

**3. Smart File Reading**

Adjusts strategy based on file size:

| File Size | Strategy | Tool Used |
|-----------|----------|-----------|
| < 10KB | Read fully | `read_file(path)` |
| 10KB - 100KB | Read with limit | `read_file(path, limit=100)` |
| 100KB - 1MB | Peek at start | `read_file(path, limit=50)` |
| > 1MB | Metadata only | Just note size and type |

**4. Data Analysis**

For data files (CSV, Excel, Parquet, HDF5, JSON):

```python
# Agent writes analysis script
analysis_script = """
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Analyze
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Dtypes:\\n{df.dtypes}")
print(f"\\nHead:\\n{df.head()}")
print(f"\\nDescribe:\\n{df.describe()}")
print(f"\\nMissing values:\\n{df.isnull().sum()}")
"""

# Executes via execute_command
# Captures and includes output in report
```

**5. Report Generation**

Generates structured markdown:

```markdown
# Directory Exploration Report: project_name

## Overview
- Total files: 150
- Total directories: 25
- Total size: 45.2 MB
- Explored on: 2025-10-10

## Directory Structure
[Hierarchical tree view]

## Files by Category

### Source Code
[File listings with brief descriptions]

### Data Files
[Detailed analysis for each with pandas output]

### Configuration
[Key configs and settings]

### Documentation
[List with summaries]

### Other Files
[Binary files, etc.]

## Summary
[Key insights and observations]
```

### Supported Data Formats

| Format | Extension | Pandas Reader |
|--------|-----------|---------------|
| CSV/TSV | .csv, .tsv | `pd.read_csv()` |
| Excel | .xlsx, .xls | `pd.read_excel()` |
| Parquet | .parquet | `pd.read_parquet()` |
| HDF5 | .h5, .hdf5 | `pd.read_hdf()` |
| JSON | .json, .jsonl | `pd.read_json()` |
| Feather | .feather | `pd.read_feather()` |

### Example Report Structure

```markdown
# Directory Exploration Report: data_pipeline

## Overview
- Total files: 47
- Total directories: 8
- Total size: 12.5 MB
- Explored on: 2025-10-10 15:30:45

## Directory Structure

```
data_pipeline/
├── src/
│   ├── ingest/
│   │   ├── extract.py (2.3 KB)
│   │   └── transform.py (4.1 KB)
│   └── analysis/
│       └── stats.py (3.2 KB)
├── data/
│   ├── raw/
│   │   └── sales.csv (450 KB)
│   └── processed/
│       └── sales_clean.parquet (280 KB)
└── config/
    └── pipeline.yaml (1.2 KB)
```

## Files by Category

### Source Code (3 files)

**src/ingest/extract.py** (line count: 85)
- Defines `DataExtractor` class
- Connects to PostgreSQL database
- Extracts sales data using SQL queries

**src/ingest/transform.py** (line count: 120)
- Implements data cleaning pipeline
- Handles missing values and outliers
- Standardizes date formats

[...]

### Data Files (2 files)

**data/raw/sales.csv**

Analysis results:
```
Shape: (1500, 8)
Columns: ['date', 'product_id', 'quantity', 'price', 'customer_id', 'region', 'category', 'revenue']

Dtypes:
date           object
product_id      int64
quantity        int64
price         float64
[...]

Missing values:
date          0
product_id    0
price        12
customer_id   5
[...]
```

[...]

## Summary

This data pipeline processes sales data from a PostgreSQL database:

1. **Data Ingestion**: `extract.py` pulls raw data, `transform.py` cleans and validates
2. **Data Quality**: 12 missing prices and 5 missing customer IDs identified in raw data
3. **Processing Flow**: Raw CSV → Cleaning → Parquet format for efficient storage
4. **Configuration**: Pipeline settings in `pipeline.yaml` specify database credentials and batch size

**Recommendations**:
- Address missing values in price and customer_id columns
- Consider adding data quality tests
- Document transformation logic in transform.py

[End of report]
```

### Example Usage

```python
# Simple exploration
from alfredo.prebuilt import ExplorationAgent

agent = ExplorationAgent(cwd="./my_project")
report = agent.explore()

# Focused exploration
agent = ExplorationAgent(
    cwd="./codebase",
    context_prompt="Focus on test coverage and documentation quality",
    model_name="gpt-4.1-mini"
)
report = agent.explore()

# Custom output location
agent = ExplorationAgent(
    cwd="./data",
    output_path="./analysis/data_exploration.md",
    preview_lines=100  # Use line-based preview
)
report = agent.explore()
```

### Dependencies

```bash
# Required for data analysis
uv add pandas

# Optional for specific formats
uv add openpyxl    # Excel support
uv add pyarrow     # Parquet support
uv add tables      # HDF5 support
```

---

## ReflexionAgent

**Purpose**: Research agent with iterative self-critique and revision using web search.

The ReflexionAgent implements the Reflexion pattern: draft an answer, critically reflect on gaps, search for information, revise with citations, and repeat until a comprehensive answer is produced.

### Features

- **Self-Critique** - Rigorous evaluation of answers for completeness and accuracy
- **Targeted Search** - Generates specific queries to address identified gaps
- **Citation Integration** - Incorporates sources with proper citations
- **Iterative Refinement** - Improves answers over multiple iterations
- **Summary Generation** - Optional polished markdown report using Agent class

### Basic Usage

```python
from alfredo.prebuilt import ReflexionAgent

# Create agent
agent = ReflexionAgent(
    model_name="gpt-4.1-mini",
    max_iterations=2
)

# Research a question
answer = agent.research("How can small businesses leverage AI to grow?")
print(answer)

# View execution trace
agent.display_trace()
```

### With Custom Output

```python
agent = ReflexionAgent(
    model_name="gpt-4.1-mini",
    max_iterations=3,
    output_path="./reports/ai_research.md",
    use_summary_agent=True,  # Generate polished summary
    verbose=True
)

answer = agent.research("What are the latest developments in quantum computing?")

# Trace shows:
# 1. Draft with self-reflection
# 2. Search queries executed
# 3. Revisions with citations
# 4. Summary agent work
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cwd` | `"."` | Working directory |
| `model_name` | `"gpt-4.1-mini"` | Model to use |
| `max_iterations` | `2` | Maximum revision cycles |
| `search_tool` | `None` | Custom search tool (default: TavilySearch) |
| `output_path` | `None` | Report save path (default: `notes/reflexion_research.md`) |
| `use_summary_agent` | `True` | Use Agent to write polished summary |
| `verbose` | `True` | Print progress updates |

### How It Works

**The Reflexion Loop**:

```
1. DRAFT
   ↓
2. REFLECT (identify gaps)
   ↓
3. SEARCH (gather information)
   ↓
4. REVISE (integrate with citations)
   ↓
5. REPEAT (until max_iterations)
```

**Graph Structure**:

```
START → draft → execute_tools → revisor
                      ↑              ↓
                      └──────────────┘
                                     ↓
                                    END
```

**State Management**:

```python
class ReflexionState(TypedDict):
    messages: Sequence[BaseMessage]  # Conversation history
    question: str                     # Research question
    iteration: int                    # Current iteration
    max_iterations: int               # Maximum iterations
```

### Detailed Workflow

**1. Draft Node**

Creates initial answer with self-reflection:

```python
# Agent generates:
{
    "answer": "~250 word draft answer",
    "reflection": {
        "missing": "What critical information is missing?",
        "superfluous": "What could be removed or simplified?"
    },
    "search_queries": [
        "specific query 1 addressing gap",
        "specific query 2 for verification",
        "specific query 3 for depth"
    ]
}
```

**2. Execute Tools Node**

Runs search queries:

```python
# For each query:
results = search_tool.invoke(query)

# Returns:
{
    "query 1": [
        {"url": "https://source1.com", "title": "...", "content": "..."},
        {"url": "https://source2.com", "title": "...", "content": "..."}
    ],
    "query 2": [...]
}
```

**3. Revisor Node**

Revises answer using search results:

```python
# Agent generates:
{
    "answer": "~250 word revised answer with inline citations [1], [2]",
    "reflection": {
        "missing": "What remains missing after incorporating new info?",
        "superfluous": "Any redundant or less relevant content?"
    },
    "search_queries": ["new query 1", "new query 2"],  # For next iteration
    "references": [
        "[1] https://actual-url-from-search-result-1.com",
        "[2] https://actual-url-from-search-result-2.com"
    ]
}
```

**4. Repeat**

Continues until `max_iterations` reached.

### Reflexion Prompts

The agent uses rich, detailed prompts for quality output:

**Draft Prompt** (key excerpts):

```
You are an expert researcher with a critical eye for quality and completeness.

1. Draft Answer (~250 words)
   - Use clear, concise language
   - Structure information logically
   - Make claims that can be verified

2. Critical Self-Reflection
   - Are all facts correct and verifiable?
   - What critical information is missing?
   - What assumptions are implicit?
   - What deeper questions does this raise?

   Be specific: Instead of "incomplete information", say "missing data
   on market size in developing countries"

3. Targeted Search Queries (1-3)
   - Queries that fill factual gaps
   - Queries that provide missing perspectives
   - Queries that verify questionable claims
```

**Revisor Prompt** (key excerpts):

```
Revise your previous answer by synthesizing new information.

Using Search Results:
- Parse JSON search results from ToolMessage
- Extract actual 'url' field from each result
- Format references: "[1] https://url, [2] https://url"

Integration Guidelines:
- Synthesize, don't just append
- Cite inline with [1], [2] notation
- Extract actual URLs from search results
- Verify claims with sources
- Remove redundancy

Quality Check:
- Have I filled the identified gaps?
- Are citations integrated naturally?
- Have I extracted actual URLs?
```

### Search Tool Integration

**Default: TavilySearch**

```python
# Automatically used if no search_tool provided
agent = ReflexionAgent()  # Uses TavilySearch

# Requires:
uv add langchain-tavily
# And TAVILY_API_KEY environment variable
```

**Custom Search Tool**:

```python
from langchain_community.tools import DuckDuckGoSearchRun

# Use custom search tool
custom_search = DuckDuckGoSearchRun(max_results=5)

agent = ReflexionAgent(
    search_tool=custom_search,
    max_iterations=2
)

answer = agent.research("Your question here")
```

### Summary Generation

With `use_summary_agent=True` (default), a secondary Agent writes a polished report:

**Summary Report Structure**:

```markdown
# Reflexion Research Report

## Question
[Research question clearly stated]

## Executive Summary
[2-3 sentence high-level summary]

## Key Findings
- Finding 1
- Finding 2
- Finding 3

## Detailed Analysis
[Full answer with proper markdown formatting]
[Headers, bullet points, inline citations [1], [2]]

## Methodology
- Model: gpt-4.1-mini
- Iterations: 2
- Approach: Reflexion with self-critique and revision

## References
- [1] https://source1.com
- [2] https://source2.com

---
*Generated by ReflexionAgent using iterative self-critique*
```

### Example Execution

```python
from alfredo.prebuilt import ReflexionAgent

# Create agent
agent = ReflexionAgent(
    model_name="gpt-4.1-mini",
    max_iterations=2,
    use_summary_agent=True,
    verbose=True
)

# Research
answer = agent.research("""
What are the key challenges and opportunities for small businesses
adopting AI in 2025?
""")

# Output shows:
# 🔍 Starting reflexion research...
# 📊 Max iterations: 2
#
# [Iteration 0]
# 🎯 DRAFT
#   Answer: [250 words on AI adoption challenges]
#   Reflection:
#     - Missing: Specific cost data for SMBs
#     - Missing: Case studies of successful adoption
#     - Superfluous: Generic benefits already well-known
#   Search Queries:
#     - "SMB AI adoption costs 2025 statistics"
#     - "small business AI implementation case studies"
#
# 🔍 SEARCH RESULTS
#   Queries executed: 2
#
# ✏️  REVISION
#   Answer: [Revised with inline citations [1], [2], [3]]
#   Reflection:
#     - Missing: Regional differences in adoption
#     - Better coverage of costs and ROI
#   Search Queries:
#     - "AI adoption regional differences small business"
#
# [Iteration 1]
# ...
#
# ✅ Research completed!
# 📝 Iterations used: 2/2
# 📝 Generating polished summary with Agent...
# 💾 Summary written to: notes/reflexion_research.md
```

### Viewing Execution Trace

```python
# Display full reflexion trace
agent.display_trace()

# Shows:
# - Question
# - Each iteration with draft/revision
# - Search results
# - Reflections
# - Final answer
# - Summary agent work (if enabled)
```

### Example Output

```python
agent = ReflexionAgent(max_iterations=1)
answer = agent.research("What is prompt engineering?")

print(answer)
```

**Output**:

```
Prompt engineering is the practice of designing and optimizing input prompts
to achieve desired outputs from large language models (LLMs). It involves
crafting clear, specific instructions that guide the model's behavior, often
incorporating techniques like few-shot learning [1], chain-of-thought reasoning [2],
and role-based prompting [3].

Key principles include:
- Clarity: Using precise, unambiguous language
- Context: Providing relevant background information
- Structure: Organizing prompts logically
- Iteration: Refining prompts based on outputs

Applications span content generation, data extraction, code synthesis, and
question-answering systems. Recent research [4] shows that systematic prompt
optimization can improve model performance by 20-50% on specific tasks,
making it a critical skill for AI practitioners.

## References
- [1] https://arxiv.org/abs/2005.14165
- [2] https://arxiv.org/abs/2201.11903
- [3] https://platform.openai.com/docs/guides/prompt-engineering
- [4] https://proceedings.neurips.cc/paper/2023/file/promptopt.pdf
```

### Dependencies

```bash
# Required
uv add langchain-tavily  # For web search
# Set TAVILY_API_KEY environment variable

# Or use custom search tool (no dependency)
```

### Best Practices

**1. Set Appropriate Iterations**

```python
# Simple questions
agent = ReflexionAgent(max_iterations=1)

# Complex research
agent = ReflexionAgent(max_iterations=3)

# Deep dive
agent = ReflexionAgent(max_iterations=5)
```

**2. Use Specific Questions**

```python
# Good - specific and answerable
"What are the top 3 programming languages for data science in 2025?"

# Bad - too broad
"Tell me about programming"
```

**3. Enable Summary for Reports**

```python
# For one-off research
agent = ReflexionAgent(use_summary_agent=False)

# For shareable reports
agent = ReflexionAgent(
    use_summary_agent=True,
    output_path="./reports/research.md"
)
```

**4. Choose Appropriate Model**

```python
# For cost-effective research
agent = ReflexionAgent(model_name="gpt-4.1-mini")

# For high-quality research
agent = ReflexionAgent(model_name="gpt-4o")

# For very long context
agent = ReflexionAgent(model_name="anthropic/claude-3-5-sonnet-20241022")
```

---

## Creating Custom Prebuilt Agents

You can create your own specialized agents by wrapping the base Agent class:

```python
from alfredo import Agent
from typing import Any

class CustomAgent:
    """Your custom specialized agent."""

    def __init__(self, cwd: str = ".", **kwargs: Any):
        # Create underlying agent
        self.agent = Agent(
            cwd=cwd,
            enable_planning=False,  # Or True, depending on use case
            **kwargs
        )

        # Set custom prompts
        self.agent.set_agent_prompt("""
Your custom system prompt here.

Task: {task}
Plan: {plan}

{tool_instructions}
        """)

    def execute_custom_task(self, task: str) -> str:
        """Execute your custom task type."""
        result = self.agent.run(task)
        return result["final_answer"]

    def display_trace(self) -> None:
        """Proxy to underlying agent's trace."""
        self.agent.display_trace()
```

**Usage**:

```python
custom_agent = CustomAgent(cwd=".")
result = custom_agent.execute_custom_task("Do something specialized")
custom_agent.display_trace()
```

## Related Documentation

- **[Agent Architecture](agent-architecture.md)** - Understanding the base Agent class
- **[Tools](tools.md)** - Available tools for agents
- **[AlfredoTool](alfredo-tools.md)** - Customizing system prompts
