import pandas as pd
import os

import faiss
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter

import pickle
import argparse

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://ndis-chatbot.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "eeee342cc23e46538d4f8d7cf40682a4"

if False:
    #reading the site content from csv to a list        
    pages = pd.read_csv('docs/NDIS_site_content.csv')

    #splitting docs into chunks
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap = 50, separator=" ")
    docs, metadatas = [], []
    for i in range(len(pages)):
        doc = text_splitter.split_text(pages.iloc[i,:]['text'])
        docs.extend(doc)
        metadatas.extend([{"source": pages.iloc[i,:]['source']}] * len(doc))

    #storing the docs into embeddings
    embeddings = OpenAIEmbeddings(chunk_size = 1)#deployment="text-embedding-ada-002", model="text-embedding-ada-002")
    store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)

if True:
    parser = argparse.ArgumentParser(description='Q&A')
    parser.add_argument('question', type=str, help='Your question for whats on City of Sydney')
    args = parser.parse_args()

    with open("faiss_store.pkl", "rb") as f:
        store = pickle.load(f)

    llm = AzureOpenAI(temperature=0, model_name = "gpt-35-turbo", deployment_name="gpt-35-turbo")

    chain = VectorDBQAWithSourcesChain.from_llm(llm=llm, vectorstore=store)
    result = chain({"question": args.question})

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")