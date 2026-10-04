from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

class MovieInfo(BaseModel):
    title: str
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    writers: Optional[str]
    producers: Optional[str]
    budget: Optional[str]
    box_office: Optional[str]
    release_date: Optional[str]
    country: Optional[str]
    summary: str

parser = PydanticOutputParser(pydantic_object=MovieInfo)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert movie information extraction assistant.
Your task is to extract useful information from a movie description provided by the user.
Extract the following information:
 {format_instructions}
    """),
    ("human", "{provided_description}"),
])

input_message = input("Enter the movie description: ")

prompt_input = prompt.invoke({
    "provided_description": input_message,
    "format_instructions": parser.get_format_instructions(),
})

response = model.invoke(prompt_input)

movie_info = parser.parse(response.content)
print("Extracted Movie Information:")
print(movie_info.model_dump_json(indent=2))
