from langchain.agents import create_agent
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

model_id = "Qwen/Qwen3-1.7B"

# 1. Initialize Tools
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)
search_tool = DuckDuckGoSearchRun()
tools = [search_tool, wiki_tool]

# 2. Setup Local Model (Qwen 3 1.7B)
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 1024, "temperature": 0.1},
)
model = ChatHuggingFace(llm=llm)

# 3. Define System Prompt
SYSTEM_PROMPT = """You are a helpful research assistant.
- Use 'duckduckgo_search' as main tool.
- Use 'wikipedia' for general knowledge and historical information.
Always provide a concise summary of your findings."""

# 4. Create the Agent
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver() # Built-in persistence for the chatbot
)

# 5. Run the Chatbot Loop
config = {"configurable": {"thread_id": "research_session_01"}}

while True:
    user_msg = input("User: ")
    if user_msg.lower() in ["exit", "quit"]: break
    
    # We use .invoke() just like before, but it's now running the new v1 graph
    result = agent.invoke(
        {"messages": [("user", user_msg)]},
        config=config
    )
    
    # The final message in the state is the agent's response
    print(f"AI: {result['messages'][-1].content}\n")