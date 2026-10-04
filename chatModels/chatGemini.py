from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("google_genai:gemini-3.8-flash")

print("Chat model initialized. Type 'exit' to quit.")
while True:
    propmt = input("You: ")
    if propmt.lower() == "exit":
        break
    response  = model.invoke(propmt)
    print("Bot: ", response.content)

