from talk2data.embedding_models.base import BaseEmbeddingModel
from talk2data.vector_databases.base import BaseVectorDatabase


class BaseSource:
    def __init__(self):
        pass

    def add_data(
        self,
        source: str,
        llm_model: BaseEmbeddingModel,
        vector_database: BaseVectorDatabase,
    ) -> None:
        raise NotImplementedError
