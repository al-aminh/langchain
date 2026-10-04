from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(model="openai/gpt-oss-120b")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert movie information extraction assistant.

Your task is to extract useful information from a movie description provided
by the user.

Extract the following information:

Title:
Genre:
Director:
Cast:
Writers:
Producers:
Budget:
Box Office:
Release Date:
Country:
Language:
Runtime:
Rating:
Plot:
Summary:
Themes:
Awards:
Production Company:
Distributor:

Rules:
- Extract information ONLY from the user's movie description.
- Do not use your own knowledge to fill in missing information.
- Do not guess or assume anything.
- If information is not mentioned, write "Not mentioned".
- If multiple values exist, separate them with commas.
- Keep names, dates, ratings, and monetary values as they appear in the input.
- The Plot should contain the detailed story information available in the description.
- The Summary should be a short summary based only on the provided description.
- Do not add explanations before or after the extracted information.
- Do not return JSON.
- Do not use Markdown tables.
- Follow the exact output format below.

Output format:

Title: <title>
Genre: <genre>
Director: <director>
Cast: <cast>
Writers: <writers>
Producers: <producers>
Budget: <budget>
Box Office: <box_office>
Release Date: <release_date>
Country: <country>
Language: <language>
Runtime: <runtime>
Rating: <rating>
Plot: <plot>
Summary: <summary>
Themes: <themes>
Awards: <awards>
Production Company: <production_company>
Distributor: <distributor>
"""
    ),
    (
        "human",
        """
Extract the movie information from the following description:

{movie_description}
"""
    )
])



movie_description = input("Please enter the movie description: ")

response = llm.invoke(prompt.format(movie_description=movie_description))
print("Extracted Movie Information:\n")
print(response.content)