# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
import operator



class AgentState(TypedDict):
    objective: str
    problem: str
    plan: str
    hypothesis: dict[str, list[str]]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    no_of_iterations: int

