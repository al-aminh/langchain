from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
model = ChatGroq(model ="openai/gpt-oss-120b")
# model = init_chat_model("openai/gpt-oss-120b", model_provider="groq")

messages = [
    SystemMessage(content="You are a helpful assistant."),
]

print("Chat model initialized. Type 'exit' to quit.")
while True:

    propmt = input("You: ")
    messages.append(HumanMessage(content=propmt))
    if propmt.lower() == "exit":
        break
    response  = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot: ", response.content)