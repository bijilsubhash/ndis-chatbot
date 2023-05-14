from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain

import csv
import pickle

import openai
import os
from dotenv import load_dotenv # Add
load_dotenv() # Add
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
print(openai.api_key)

pages = []

#reading the site content from csv to a list
with open('docs/NDIS_site_content.csv', 'r') as file:
    reader = csv.reader(file, delimiter='\n')
    for row in reader:
        pages.append(row[0])

#splitting docs into chunks
text_splitter = CharacterTextSplitter(chunk_size=3000, separator="\n")
docs = []
for page in pages:
    doc = text_splitter.split_text(page)
    docs.extend(doc)

# #storing the docs into embeddings
# store = FAISS.from_texts(docs, OpenAIEmbeddings())

# with open("faiss_store.pkl", "wb") as f:
#     pickle.dump(store, f)

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

template = """Given the following chat history 
and a follow up question, rephrase the follow up input question to be a standalone question.
Or end the conversation if it seems like it's done.

Chat History:\"""
{chat_history}
\"""
Follow Up Input: \"""
{question}
\"""
Standalone question:"""
 
condense_question_prompt = PromptTemplate.from_template(template)
 
template = """You are a friendly, conversational chatbot. 
Use the context in vector store, help find what they want, and answer any questions.
If the question cannot be answered using the information provided answer with "I don't know"

Context:\"""
 
{context}
\"""
Question:\"
\"""
 
Helpful Answer:"""
 
qa_prompt= PromptTemplate.from_template(template)

# define two LLM models from OpenAI
llm = OpenAI(temperature=0)
 
streaming_llm = OpenAI(
    streaming=True,
    callback_manager=CallbackManager([
        StreamingStdOutCallbackHandler()
    ]),    verbose=True,
    max_tokens=500,
    temperature=0.2
)
 
# use the LLM Chain to create a question creation chain
question_generator = LLMChain(
    llm=llm,
    prompt=condense_question_prompt
)
 
# use the streaming LLM to create a question answering chain
doc_chain = load_qa_chain(
    llm=streaming_llm,
    chain_type="stuff",
    prompt=qa_prompt
)

chatbot = ConversationalRetrievalChain(
    retriever=store.as_retriever(),
    combine_docs_chain=doc_chain,
    question_generator=question_generator
)

# create a chat history buffer
chat_history = []
# gather user input for the first question to kick off
question = input('Hi there, I am a chatbot who can access the NDIS webpages for you. What can I help you with today?\n\n')
 
# keep the bot running in a loop to simulate a conversation
while True:
    result = chatbot({"question": question, "chat_history": chat_history})
    print("\n")
    chat_history.append((result["question"], result["answer"]))
    question = input()