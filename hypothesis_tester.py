from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from state import AgentState

def hypothesis_tester_agent(state:AgentState)->AgentState:
    llm = ChatOllama(model="llama3.1")
    # tools = [TavilySearchResults(max_results=5)]
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert problem solver tasked with evaluating a given hypothesis."
            " Your role is to assess the hypothesis and provide a detailed solution based on it."
            " When presented with a hypothesis and problem, you will:\n"
            "   1. Carefully analyze the hypothesis in the context of the given problem.\n"
            "   2. Determine if the hypothesis is correct or incorrect based on your analysis.\n"
            "   3. If the hypothesis is correct, develop a comprehensive solution to the problem.\n"
            "   4. If the hypothesis is incorrect, explain why and propose an alternative approach.\n"
            "   5. Break down your solution or alternative approach into clear, actionable steps.\n"
            "   6. Consider potential challenges and include strategies to overcome them.\n"
            "   8. Organize your response in a logical sequence, ensuring each point builds upon the previous.\n"
            "   9. Provide a clear, concise summary of your evaluation and proposed solution.\n"
            " Remember, your goal is to critically assess the hypothesis and formulate an effective"
            " solution to the problem, guiding the user through your reasoning and approach."
            " And you are all by yourself, you cannot ask question to the user"
        ),
        ("user", " Here is the problem that needs to be solved {problem} and here is the hypothesis {hypothesis}"),
        ]
    )

    final_result = ""
    for hypothesis in state["hypothesis"]:
        tester_agent = prompt | llm | parser #| llm.bind_tools(tools)
        result =  tester_agent.invoke({"problem": state["problem"], "hypothesis": hypothesis})
        final_result += result

    # Write the result to a file
        with open('result.txt', 'a') as f:
            f.write(f"{state["no_of_iterations"]} Hypothesis Tester Agent : {result}\n\n")
            f.write("_"*15)

    return { "messages": [HumanMessage(content=final_result)], "no_of_iterations": state["no_of_iterations"] + 1 }



