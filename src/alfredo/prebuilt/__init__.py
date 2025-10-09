"""Pre-built agents for common tasks.

This module provides ready-to-use specialized agents for common tasks:
- ExplorationAgent: Explore directories and generate markdown reports
- ReflexionAgent: Research questions with iterative self-critique and revision
"""

from alfredo.prebuilt.explore import ExplorationAgent
from alfredo.prebuilt.reflexion import ReflexionAgent

__all__ = ["ExplorationAgent", "ReflexionAgent"]
