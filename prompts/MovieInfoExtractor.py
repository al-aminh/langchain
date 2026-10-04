import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Pydantic Schema
# --------------------------------------------------

class MovieInfo(BaseModel):
    title: str
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    writers: Optional[str] = None
    producers: Optional[str] = None
    budget: Optional[str] = None
    box_office: Optional[str] = None
    release_date: Optional[str] = None
    country: Optional[str] = None
    summary: str


# --------------------------------------------------
# LangChain Setup
# --------------------------------------------------

@st.cache_resource
def get_model():
    return ChatGroq(
        model="openai/gpt-oss-120b"
    )


model = get_model()

parser = PydanticOutputParser(
    pydantic_object=MovieInfo
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert movie information extraction assistant.

        Your task is to extract useful information from a movie
        description provided by the user.

        Extract the following information:

        {format_instructions}

        Important instructions:
        - Extract only information that is present in the description.
        - Do not invent or hallucinate information.
        - If an optional field is not available, return null.
        - Return the output according to the required format.
        """
    ),
    (
        "human",
        "{provided_description}"
    ),
])


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888;
        font-size: 18px;
        margin-bottom: 35px;
    }

    .movie-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #1e1e1e;
        border: 1px solid #333;
        margin-bottom: 15px;
    }

    .field-title {
        color: #aaa;
        font-size: 14px;
        margin-bottom: 3px;
    }

    .field-value {
        font-size: 17px;
        font-weight: 500;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎬 Movie Information Extractor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Extract structured movie information using AI'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("Movie Description")

description = st.text_area(
    "Paste a movie description below",
    height=250,
    placeholder=(
        "Example:\n\n"
        "Inception is a science-fiction thriller directed by "
        "Christopher Nolan and released in 2010..."
    ),
    label_visibility="collapsed"
)


# --------------------------------------------------
# Extract Button
# --------------------------------------------------

extract_button = st.button(
    "🎯 Extract Movie Information",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# Movie Extraction
# --------------------------------------------------

if extract_button:

    if not description.strip():
        st.warning("Please enter a movie description first.")

    else:

        try:
            with st.spinner("🤖 Extracting movie information..."):

                prompt_input = prompt.invoke({
                    "provided_description": description,
                    "format_instructions": parser.get_format_instructions(),
                })

                response = model.invoke(prompt_input)

                movie_info = parser.parse(response.content)

            st.success("Movie information extracted successfully!")

            # --------------------------------------------------
            # Display Results
            # --------------------------------------------------

            st.divider()

            st.header("🎬 Extracted Movie Information")

            # Title
            st.subheader(movie_info.title)

            # First row
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Genre**")
                st.write(
                    ", ".join(movie_info.genre)
                    if movie_info.genre
                    else "Not available"
                )

            with col2:
                st.markdown("**Director**")
                st.write(
                    movie_info.director
                    if movie_info.director
                    else "Not available"
                )

            with col3:
                st.markdown("**Release Date**")
                st.write(
                    movie_info.release_date
                    if movie_info.release_date
                    else "Not available"
                )

            # Second row
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Country**")
                st.write(
                    movie_info.country
                    if movie_info.country
                    else "Not available"
                )

            with col2:
                st.markdown("**Budget**")
                st.write(
                    movie_info.budget
                    if movie_info.budget
                    else "Not available"
                )

            with col3:
                st.markdown("**Box Office**")
                st.write(
                    movie_info.box_office
                    if movie_info.box_office
                    else "Not available"
                )

            # Cast
            st.markdown("### 👥 Cast")

            if movie_info.cast:
                for actor in movie_info.cast:
                    st.markdown(f"- {actor}")
            else:
                st.write("Not available")

            # Writers
            st.markdown("### ✍️ Writers")
            st.write(
                movie_info.writers
                if movie_info.writers
                else "Not available"
            )

            # Producers
            st.markdown("### 🎥 Producers")
            st.write(
                movie_info.producers
                if movie_info.producers
                else "Not available"
            )

            # Summary
            st.markdown("### 📝 Summary")
            st.info(movie_info.summary)

            # --------------------------------------------------
            # Raw JSON
            # --------------------------------------------------

            with st.expander("View Raw JSON"):

                st.json(
                    movie_info.model_dump()
                )

        except Exception as e:

            st.error(
                "Something went wrong while extracting "
                "the movie information."
            )

            with st.expander("Error details"):
                st.exception(e)
