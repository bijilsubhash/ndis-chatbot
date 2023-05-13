
"""
This script uses retrieval chain to answer questions with and without memory. 
Pinecone was used to store the vectors in a database in the cloud.
Parameters like text splitter type, model type, search type etc could be tuned to optimize the results even further. 

query #1, #2 and #3 are without memory
query #4, #5 and #6 are with memory 

"""

import os
import sys
import csv

from langchain.document_loaders import CSVLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

import openai
import pinecone

#configurations
csv.field_size_limit(sys.maxsize)
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

#loading the data
loader = CSVLoader(file_path="docs/NDIS_site_content.csv", csv_args={"delimiter": "\n"})
data = loader.load()

#split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

#select which embeddings we want to use
embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")

# initialize pinecone
pinecone.init(api_key=PINECONE_API_KEY,  environment=PINECONE_ENV)
index_name = "ndis-chatbot"

# create the vectorstore to use as the index --- DONE 
# db = Pinecone.from_documents(texts, embeddings, index_name=index_name)

#if you already have an index, you can load it like this --- USE THIS
db = Pinecone.from_existing_index(index_name, embeddings)

#expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_template = """You are a conversational NDIS expert with access to NDIS context. 
You are informative and provides details from the context. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#create a chain to answer questions
chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(ChatOpenAI(model_name="gpt-3.5-turbo"), 
                                 chain_type="stuff", retriever=retriever, 
                                 chain_type_kwargs=chain_type_kwargs)

#query #1
query = "what is NDIS?"
result = qa({"query": query})
print("query #1")
print(result)

#query #2
query = "how can I get NDIS funding?"
result = qa({"query": query})
print("query #2")
print(result)

#query #3
query = "why is my NDIS funding application very long?"
result = qa({"query": query})
print("query #3")
print(result)

#create a chain to answer questions using conversational retrieval chain i.e. with memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever)

#query #4
chat_history = []
query = "what is NDIS?"
result = qa({"question": query, "chat_history": chat_history})

print("query #4")
print(result["answer"], chat_history)

#query #5
chat_history.append((query, result["answer"]))
query = "how can I get NDIS funding?"
result = qa({"question": query, "chat_history": chat_history})

print("query #5")
print(result["answer"], chat_history)

#query #6
chat_history.append((query, result["answer"]))
query = "why is my NDIS funding application very long?"
result = qa({"question": query, "chat_history": chat_history})

print("query #6")
print(result["answer"], chat_history)


"""
query #1
{'query': 'what is NDIS?', 'result': 'The NDIS (National Disability Insurance Scheme) is a national scheme in Australia that provides funding and support for people with permanent and significant disabilities. 
It is designed to help people with disabilities to access the services and support they need to lead an ordinary life. The NDIS provides funding for a wide range of services and supports, including therapy, equipment, and personal care. 
It is a government-funded scheme that aims to improve the lives of people with disabilities and their families.'}
query #2
{'query': 'how can I get NDIS funding?', 'result': "To get NDIS funding, you will need to be eligible for the scheme. 
This means you must have a permanent and significant disability that affects your ability to participate in everyday activities. 
You can check your eligibility by completing an Access Request Form on the NDIS website or by contacting the NDIS directly. 
Once you are deemed eligible, you can then develop a plan with an NDIS planner or Local Area Coordinator to determine the supports you need and how much funding you may be entitled to. 
It's important to note that the NDIS is not a welfare system and is not means-tested, so your income or assets will not affect your eligibility for funding."}
query #3
{'query': 'why is my NDIS funding application very long?', 'result': 'The NDIS funding application process can be lengthy as it involves gathering detailed information about your disability, support needs, and goals. 
This information is used to determine your eligibility for NDIS funding and the level of support you require. It is important that the application is thorough and accurate to ensure that you receive the appropriate supports and funding. 
If you are experiencing delays or have concerns about the application process, you can contact the NDIS directly for assistance.'}
query #4
The NDIS stands for the National Disability Insurance Scheme. It is a government-funded program in Australia that provides support and funding for people with permanent and significant disabilities. 
The aim of the scheme is to help people with disabilities to live more independently and to improve their access to services and support. []
query #5
To obtain NDIS funding, you need to complete an Access Request Form and submit it to the NDIS. Once your application is approved, you can then create a plan that outlines the supports you need. 
Once your plan is approved, you can start using your NDIS funding on the supports set out in your plan. [('what is NDIS?', 'The NDIS stands for the National Disability Insurance Scheme. 
It is a government-funded program in Australia that provides support and funding for people with permanent and significant disabilities. 
The aim of the scheme is to help people with disabilities to live more independently and to improve their access to services and support.')]
query #6
To obtain NDIS funding, you will need to complete an Access Request Form and apply to the NDIS. Once your application is approved, you can work with an NDIS planner to develop a plan for your supports. 
Once your plan is approved, you can start using your NDIS funding on the supports outlined in your plan. [('what is NDIS?', 'The NDIS stands for the National Disability Insurance Scheme. 
It is a government-funded program in Australia that provides support and funding for people with permanent and significant disabilities. 
The aim of the scheme is to help people with disabilities to live more independently and to improve their access to services and support.'), 
('how can I get NDIS funding?', 'To obtain NDIS funding, you need to complete an Access Request Form and submit it to the NDIS. Once your application is approved, you can then create a plan that outlines the supports you need. 
Once your plan is approved, you can start using your NDIS funding on the supports set out in your plan.')]
"""