from dotenv import load_dotenv

load_dotenv()

#  using the youtube-transcript-api package to fetch video transcript
from youtube_transcript_api import YouTubeTranscriptApi

#  now using the text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# embedding model ( each chunk will be transformed in the vectors)
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# using a vector store ( chroma in here)
from langchain_chroma import Chroma

# now we will merge the query and the retrieved vector
from langchain_core.prompts import PromptTemplate

# taking the model from the huggingface
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
)
model = ChatHuggingFace(llm=llm)


# loading the transcript
yt = YouTubeTranscriptApi()

try:
    transcript = yt.fetch("Gfr50f6ZBvo")
    clean_text = " ".join([item.text for item in transcript])
except Exception:
    clean_text = "No transcript available."


# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

# Perform the split
chunks = splitter.create_documents([clean_text], metadatas=[{}])


embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./vector_store/yt_store",
    collection_name="yt_video_transcript_collections",
)

vector_store.add_documents(chunks)


# this is the result dict in the store
# result = vector_store._collection.get(
#     ids=["6325cf6f-e29c-4ebe-a929-9dde9916065f"], include=["documents", "embeddings"]
# )

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
query = "can you tell me what is deepmind and who is ceo of deepmind"

result_from_vector = retriever.invoke(query)

# print(result[2].page_content)

prompt = PromptTemplate(
    template="give proper anwer based on this text -> {text} and the query -> {query}",
    input_variables=["text", "query"],
)

chain = prompt | model

vector_content = result_from_vector[2].page_content

result = chain.invoke({"text": vector_content, "query": query})

print(result.content)
