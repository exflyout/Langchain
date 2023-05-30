# 필요한 패키지를 설치합니다.
# 터미널에서 다음 명령을 실행하세요.
# pip install openai langchain google-search-results wikipedia faiss-cpu sentence_transformers tiktoken pypdf

# API 키를 설정합니다.
import os

# API 키를 저장한 파일을 읽습니다.
with open('api_keys.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    key, value = line.strip().split('=')
    os.environ[key] = value

# langchain 라이브러리의 다양한 모듈을 임포트합니다.
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain import ConversationChain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
import langchain

# ChatOpenAI 객체를 생성합니다. 이 객체는 GPT-3.5-turbo 모델을 사용하여 채팅을 수행합니다.
chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

# PyPDFLoader를 사용하여 PDF 문서를 로드하고, 텍스트를 분할합니다.
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("/path/to/your/pdf") # 여기에 PDF 파일의 경로를 입력하세요.
documents = loader.load_and_split()

# 텍스트 분할
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 요약 작업을 수행합니다.
from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(chat, chain_type="map_reduce", verbose=True)
chain.run(docs)

# 임베딩을 생성합니다.
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings# OpenAI 임베딩을 사용하려면 이 줄의 주석을 해제하세요.
embeddings = HuggingFaceEmbeddings() # HuggingFace 임베딩을 사용합니다.

# 임베딩을 FAISS 벡터스토어에 저장합니다.
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
index = VectorstoreIndexCreator(
    vectorstore_cls=FAISS,
    embedding=embeddings,
    ).from_loaders([loader])

# 파일로 저장합니다.
index.vectorstore.save_local("/path/to/save/vectorstore") # 여기에 벡터스토어를 저장할 경로를 입력하세요.
