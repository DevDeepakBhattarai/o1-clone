from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser

from state import AgentState


def expander_agent(state:AgentState):   
    llm = ChatOllama(model="llama3.1")
    parser = StrOutputParser()
    # tools = [TavilySearchResults(max_results=5)]
    prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an AI agent designed to enhance and refine questions."
            " Your primary function is to take a given question and improve it by adding context, specificity, and depth."
            " This process should make the question more insightful, precise, and likely to elicit more comprehensive and valuable answers."
            " When presented with a question, you will: \n"
            " 1. Carefully analyze the original question to understand its core topic or subject. and determine the implicit goal of the question.\n"
            " 2. Add more context to the question to make it more specific and detailed.\n"
            " 3. Improve the clarity and structure of the question to make it more engaging, informative, and readable.\n\n"
            "But remember, your goal is not to answer the question, but to refine and improve it."
            " The enhanced question should maintain the original intent and all the original information, while being more likely to generate insightful, comprehensive, and valuable responses."
            
        ),
        ("user", "Here is the question that needs to be improved {objective}"),
        ]
    )

    print(state["objective"])
    expander_agent = prompt | llm | parser # | llm.bind_tools(tools)
    result =  expander_agent.invoke(state)

    # Write the result to a file
    with open('result.txt', 'a') as f:
        f.write(f"{state["no_of_iterations"]} Expander Agent : {result}\n\n")
        f.write("_"*15)
        
    return { "problem": result, "no_of_iterations": state["no_of_iterations"] + 1 }



