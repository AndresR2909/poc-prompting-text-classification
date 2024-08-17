import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class LlmServiceModel:

    def __init__(self, model_deployment: str, api_version: str = None):
        self.api_version = api_version
        self.model_deployment = model_deployment
        if model_deployment == "gpt-4o-mini":
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
            self.llm = self._get_openia_client()
        if model_deployment == "gpt-35-proyecto1":
            AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
            AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
            # AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY2")
            # AZURE_OPENAI_ENDPOINT =  os.environ.get("AZURE_OPENAI_ENDPOINT2")
            self.llm = self._get_azure_client()

    def _get_azure_client(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=self.model_deployment,
            api_version=self.api_version,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return self.llm

    def _get_openia_client(self):
        self.llm = ChatOpenAI(
            model=self.model_deployment,
            temperature=0,
            max_tokens=20,
            timeout=None,
            max_retries=2,
        )
        return self.llm
