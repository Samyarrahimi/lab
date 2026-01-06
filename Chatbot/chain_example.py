from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer

# 1. Load the tokenizer separately to get the specific "Stop Token"
model_id = "Qwen/Qwen3-1.7B"
#tokenizer = AutoTokenizer.from_pretrained(model_id)

llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 500, # Keep it short to prevent long loops
        "temperature": 0.1,    # Lower temperature = less "wandering"
        "repetition_penalty": 1.1, # FIX: Prevents the "Reasoning: Reasoning:" loop
        "do_sample": True,
    },
)

# 2. Refine the Prompt 
# Small models need very direct instructions to avoid "test-taking" mode
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a scientist. Give a one-paragraph clear answer for the question.\n"),
    ("user", "{question}")
])

# 3. Create the Chain
chain = prompt | llm | StrOutputParser()

# 4. Run it
question = "What is the difference between dark matter and dark energy?\n"
# Note: Some local models return the prompt + the answer. 
# We can use .split() or a cleaner parser if needed.
response = chain.invoke({"question": question})

print("--- Answer ---")
print(response)