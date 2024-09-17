from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser

from state import AgentState


def planner_agent(state:AgentState):
    llm = ChatOllama(model="llama3.1")
    # tools = [TavilySearchResults(max_results=5)]
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert strategist tasked with formulating a detailed plan to solve a given problem."
            " Your role is to create a step-by-step guide, not to solve the problem yourself."
            " When presented with a problem, you will:"
            "   1. Carefully analyze the problem statement and its context.\n"
            "   2. Identify the specific field or domain the problem relates to.\n"
            "   3. Break down the problem into smaller, manageable components.\n"
            "   4. For each component, outline clear, actionable steps to address it.\n"
            "   5. Consider potential obstacles and include contingency steps in your plan.\n"
            "   6. If additional information is needed, use the Tavily Search tool to gather relevant data.\n"
            "   7. Organize the steps in a logical sequence, ensuring each builds upon the previous.\n"
            "   8. Provide a clear, concise summary of the overall strategy at the end of your plan.\n"
            " Remember, your goal is to create a comprehensive roadmap that guides the another AI assistant"
            " through the process of solving the problem efficiently and effectively."
        ),
        ("user", " Here is the problem that needs to be solved {problem}"),
        ]
    )

    planner_agent = prompt  | llm | parser # | llm.bind_tools(tools) 
    result =  planner_agent.invoke(state)

    # Write the result to a file
    with open('result.txt', 'a') as f:
        f.write(f"{state["no_of_iterations"]} Planner Agent : {result}\n\n")
        f.write("_"*15)

    return { "plan": result, "no_of_iterations": state["no_of_iterations"] + 1 }



