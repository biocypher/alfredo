"""Basic usage example for Alfredo agent."""

from pathlib import Path

from alfredo import Agent

# Create an agent
agent = Agent(cwd=str(Path.cwd()))

# Get the system prompt that would be sent to the model
print("=== SYSTEM PROMPT ===")
print(agent.get_system_prompt(include_examples=True))
print("\n" + "=" * 80 + "\n")

# List available tools
print("=== AVAILABLE TOOLS ===")
for tool_id in agent.get_available_tools():
    print(f"  - {tool_id}")
print("\n" + "=" * 80 + "\n")

# Simulate model responses and execute tools
print("=== TOOL EXECUTION EXAMPLES ===\n")

# Example 1: List files
print("Example 1: List files in current directory")
model_output_1 = """
Let me list the files in the current directory.

<list_files>
<path>.</path>
<recursive>false</recursive>
</list_files>
"""
result_1 = agent.execute_from_text(model_output_1)
if result_1:
    print(f"Success: {result_1.success}")
    print(f"Output:\n{result_1.output}\n")

# Example 2: Create a file
print("\nExample 2: Create a new file")
model_output_2 = """
I'll create a sample Python file.

<write_to_file>
<path>example.py</path>
<content>
def hello(name: str) -> str:
    \"\"\"Greet someone by name.\"\"\"
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(hello("World"))
</content>
</write_to_file>
"""
result_2 = agent.execute_from_text(model_output_2)
if result_2:
    print(f"Success: {result_2.success}")
    print(f"Output: {result_2.output}\n")

# Example 3: Read the file back
print("\nExample 3: Read the file back")
model_output_3 = """
Let me read the file to verify its contents.

<read_file>
<path>example.py</path>
</read_file>
"""
result_3 = agent.execute_from_text(model_output_3)
if result_3:
    print(f"Success: {result_3.success}")
    print(f"Output:\n{result_3.output}\n")

# Example 4: Search for content
print("\nExample 4: Search for 'hello' in Python files")
model_output_4 = """
<search_files>
<path>.</path>
<regex>hello</regex>
<file_pattern>*.py</file_pattern>
</search_files>
"""
result_4 = agent.execute_from_text(model_output_4)
if result_4:
    print(f"Success: {result_4.success}")
    print(f"Output:\n{result_4.output}\n")

# Clean up
print("\nCleaning up example file...")
Path("example.py").unlink(missing_ok=True)
print("Done!")
