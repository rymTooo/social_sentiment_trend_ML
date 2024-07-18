
# fixing unicode error in google colab
import locale

locale.getpreferredencoding = lambda: "UTF-8"

# Import dependencies
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
template = ChatPromptTemplate.from_messages([
    ("system", "Persona : You are a helpful AI that knows everything include expert knowledge. Your name is {name} and you are girl."),
    ("system", "Task : Your task is to assist with  answer all question and give all detailed about it."),
    ("system", "Context: You have access to all of knowledge on all thing."),
    # ("system", "Format: Answer Thai languague only"),
    ("human", "{user_input}"),
    ("ai", "Response:"),
])
llm = Ollama(
    base_url = "http://192.168.123.110:31644",
    model="qwen2:72b",
    temperature=0.1,  # Adjust temperature for diversity
) 
def generate_response(name, user_input):
    try:
        # Generate the prompt value with context and task
        prompt_value = template.invoke(
            {
                "name": name,
                "user_input": user_input
            }
        )
    except Exception as e:
        print(f"Error during template invocation: {e}")
        prompt_value = None

    if prompt_value:
        try:
            # Format the input and get the response from the LLM
            formatted_input = prompt_value['content'] if isinstance(prompt_value, dict) else prompt_value
            response = llm.invoke(formatted_input)
            return response
        except Exception as e:
            print(f"Error during LLM invocation: {e}")
            return None
name = "Malenia"
user_input = "Did you know about resin?"
response = generate_response(name, user_input)
print(response)