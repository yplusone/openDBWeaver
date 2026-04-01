"""Setup LLM configuration for LATS."""
import os
import sys

from langchain_openai import ChatOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import API_KEY, BASE_URL, EXPT_LLM_MODEL, API_KEY, BASE_URL,OPTIMIZE_LLM_MODEL,CODEGEN_LLM_MODEL
import httpx
unsafe_client = httpx.Client(
    verify=False,
    http2=False, # 禁用 HTTP/2
    timeout=httpx.Timeout(60.0, read=60.0), # 增加读取超时到120秒
    proxy="http://127.0.0.1:31328" # 显式指定代理
)
# Initialize LLM
llm = ChatOpenAI(model=EXPT_LLM_MODEL, api_key=API_KEY, base_url=BASE_URL,http_client=unsafe_client,max_retries=5)

# Initialize Advanced LLM
advanced_llm = ChatOpenAI(model=CODEGEN_LLM_MODEL, api_key=API_KEY, base_url=BASE_URL,http_client=unsafe_client,max_retries=5)

think_llm = ChatOpenAI(model=OPTIMIZE_LLM_MODEL, api_key=API_KEY, base_url=BASE_URL,http_client=unsafe_client,max_retries=5)