# 필요한 패키지를 설치합니다.
!pip install openai
!pip install langchain
!pip install google-search-results
!pip install wikipedia
!pip install faiss-cpu # Facebook과 MIT 라이선스의 오픈소스 벡터 데이터베이스
!pip install sentence_transformers # HuggingFace 임베딩 사용을 위해 필요
!pip install tiktoken # 요약 작업을 위해 필요
!pip install pypdf

# API 키를 설정합니다.
import os

# OpenAI API 키. https://platform.openai.com/account/api-keys 에서 얻을 수 있습니다.
OPENAI_API_KEY = "your_openai_api_key" 
# HuggingFace API 키. https://huggingface.co/settings/tokens 에서 얻을 수 있습니다.
HUGGINGFACEHUB_API_TOKEN = "your_huggingface_api_token" 
# SERPAPI API 키. https://serpapi.com/manage-api-key 에서 얻을 수 있습니다.
SERPAPI_API_KEY = "your_serpapi_api_key" 

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY


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
# embeddings = OpenAIEmbeddings() # OpenAI 임베딩을 사용하려면 이 줄의 주석을 해제하세요.
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


