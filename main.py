from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, START
from expander import expander_agent
from hypothesis import hypothesis_generator_agent
from hypothesis_tester import hypothesis_tester_agent
from planner import planner_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from state import AgentState

load_dotenv()


workflow = StateGraph(AgentState)

workflow.add_node("expander", expander_agent)
workflow.add_node("planner", planner_agent)
workflow.add_node("hypothesis_generator", hypothesis_generator_agent)
workflow.add_node("hypothesis_tester", hypothesis_tester_agent)

workflow.add_edge(START, "expander")
workflow.add_edge("expander", "planner")
workflow.add_edge("planner", "hypothesis_generator")
workflow.add_edge("hypothesis_generator", "hypothesis_tester")

graph = workflow.compile()


events = graph.stream(
    {
        "objective": "oyfjdnisdr rtqwainr acxz mynzbhhx -> Think step by step \n"
                     "Use the example above to decode: \n"
                     "oyekaijzdf aaptcg suaokybhai ouow aqht mynznvaatzacdfoulxxz",
        "no_of_iterations": 1
    }
)

for event in events:
    print(event)



