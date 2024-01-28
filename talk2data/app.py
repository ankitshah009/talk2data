from typing import Dict, List, Optional
import os

from talk2data.embedding_models_config import EmbeddingModelsConfig
from talk2data.enums import MEDIA_TYPE
from talk2data.llms.base import BaseLLM
from talk2data.llms.openai import OpenAi
from talk2data.sources.utils import SourceUtils
from talk2data.types import MediaData, QueryResult
from talk2data.vector_databases.base import BaseVectorDatabase
from talk2data.vector_databases.chromadb import ChromaDB


class App:
    def __init__(
            self,
            embedding_models_config: Optional[EmbeddingModelsConfig] = None,
            vector_database: Optional[BaseVectorDatabase] = None,
            llm: Optional[BaseLLM] = None,
    ):
        self.embedding_models_config = (
            embedding_models_config
            if embedding_models_config
            else EmbeddingModelsConfig()
        )
        self.vector_database = (
            vector_database
            if vector_database
            else ChromaDB(embedding_models_config=self.embedding_models_config)
        )

        self.llm = llm if llm else OpenAi(self.vector_database)
        self.source_utils = SourceUtils()

    def add_data(self, source: str):
        self.source_utils.add_data(
            source, self.embedding_models_config, self.vector_database
        )

    def query(
            self, query: str, media_types: List[MEDIA_TYPE] = [MEDIA_TYPE.IMAGE], n_results: int = 1
    ) -> QueryResult:
        data = self.get_data(query, media_types, n_results)
        response = self.llm.query(query, data)
        return response

    def get_data(
            self, query: str, media_types: List[MEDIA_TYPE] = [MEDIA_TYPE.IMAGE], n_results: int = 1
    ) -> Dict[MEDIA_TYPE, List[MediaData]]:
        return self.source_utils.get_data(
            query, media_types, self.embedding_models_config, self.vector_database, n_results
        )

    def run(self):
        import subprocess
        try:
            import streamlit
            streamlit_file_path = os.path.join(os.path.dirname(__file__), 'streamlit_app.py')
            run_process = subprocess.Popen(['streamlit', 'run', streamlit_file_path, '> NUL'])
            run_process.communicate()
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "The required dependencies for ui are not installed."
                ' Please install with `pip install --upgrade "talk2data[ui]"`'
            )
