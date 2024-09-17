from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage

from state import AgentState

def hypothesis_generator_agent(state:AgentState):
    llm = ChatOllama(model="llama3.1")
    # tools = [TavilySearchResults(max_results=5)]
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert theorist tasked with generating a list of hypotheses based on a given problem and plan."
            " Your role is to create diverse and distinct hypotheses, not to solve the problem yourself."
            " When presented with a problem and plan, you will:\n"
            "   1. Carefully analyze the problem statement, its context, and the proposed plan.\n"
            "   2. Generate multiple hypotheses that could explain or address the problem.\n"
            "   3. Ensure each hypothesis is unique and offers a different perspective.\n"
            "   4. Format your response as a JSON object with the key 'hypothesis' and a list of hypotheses as the value.\n"
            "   5. Ensure that the hypotheses are not just different versions of the same hypothesis, but genuinely distinct ideas.\n"
            "   6. Make sure that each item of the array corresponds to a hypothesis and not the continuation of the previous hypothesis."
            " One item of the array should always represent one hypothesis.\n"
            " Remember, your goal is not to answer the question but to provide a variety of theoretical approaches that another AI assistant"
            " can consider when addressing the problem. Your output should strictly follow this format:"
            " {{\"hypothesis\": [\"<Hypothesis 1>\", \"<Hypothesis 2>\", \"<Hypothesis 3>\", ...]}}"
            " Do not include any additional explanation or commentary outside of this JSON structure."
        ),
        ("user", " Here is the problem that needs to be solved {problem} and the plan that needs to be followed {plan}"),
        ]
    )

    hypothesis_agent = prompt | llm | parser #| llm.bind_tools(tools)

    def parse_output(output)-> dict[str, list[str]]:
        try:
            return parser.parse(output)
        except Exception:
            # If parsing fails, retry the LLM call
            return hypothesis_agent.invoke(state)

    result = parse_output(hypothesis_agent.invoke(state))
    while not isinstance(result, dict) or 'hypothesis' not in result:
        result = parse_output(hypothesis_agent.invoke(state))

    # Write the result to a file
    with open('result.txt', 'a') as f:
        f.write(f"{state['no_of_iterations']} Hypothesis Agent : {str(result)}\n\n")    
        f.write("_"*15)
    return { "hypothesis": result["hypothesis"], "no_of_iterations": state["no_of_iterations"] + 1 }



