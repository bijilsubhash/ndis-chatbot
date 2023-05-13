import xmltodict
import requests
from bs4 import BeautifulSoup
import csv
import os
import json
import ast

import faiss
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI, PromptTemplate
#from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter

import pickle
import argparse

openai.api_key = os.getenv("OPENAI_API_KEY")

# def extract_text_from(url):
#     html = requests.get(url).text
#     soup = BeautifulSoup(html, features="html.parser")
#     text = soup.get_text()

#     lines = (line.strip() for line in text.splitlines())
#     return '\n'.join(line for line in lines if line)

# r = requests.get("https://whatson.cityofsydney.nsw.gov.au/api/sitemap.xml")
# xml = r.text
# raw = xmltodict.parse(xml)

# pages = []

# for info in raw['urlset']['url']:
#     url = info['loc']
#     pages.append({'text': extract_text_from(url), 'source': url})

# #saving the site content to a csv
# with open('docs/site_content.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     for page in pages:
#         writer.writerow([page])

pages = []

#reading the site content from csv to a list
with open('docs/site_content.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        pages.append(row[0])

#splitting docs into chunks
text_splitter = CharacterTextSplitter(chunk_size=1500, separator=" ")
docs, metadatas = [], []
for page in pages:
    page = ast.literal_eval(page)
    doc = text_splitter.split_text(page['text'])
    docs.extend(doc)
    metadatas.extend([{"source": page['source']}] * len(doc))

#storing the docs into embeddings
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)

parser = argparse.ArgumentParser(description='Q&A')
parser.add_argument('question', type=str, help='Your question for whats on City of Sydney')
args = parser.parse_args()

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

template = """You are a chatbot having a conversation with a human. If the
question cannot be answered using the information provided answer
with "I don't know"

{summaries}
Human: {question}
Chatbot:"""


chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=store.as_retriever(),
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=template,
            input_variables=["summaries", "question"],
        ),
    },
)

result = chain({"question": args.question})

print(f"Answer: {result['answer']}")
# print(f"Sources: {result['sources']}")