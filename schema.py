from enum import Enum, EnumMeta
from typing import Union


class EnumMetaClass(Enum):

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value.upper() == other.value.upper()
        return self.value == other

    def __hash__(self):
        return hash(self._name_)

    def __str__(self):
        return self.value

    @classmethod
    def get_enum(cls, value: str) -> Union[EnumMeta, None]:
        return next(
            (
                enum_val
                for enum_val in cls
                if (enum_val.value == value)
                   or (
                           isinstance(value, str)
                           and isinstance(enum_val.value, str)
                           and (value.lower() == enum_val.value.lower() or value.upper() == enum_val.name.upper())
                   )
            ),
            None,
        )

    @classmethod
    def _missing_(cls, name):
        for member in cls:
            if isinstance(member.name, str) and isinstance(name, str) and member.name.lower() == name.lower():
                return member


class EmbeddingTypes(EnumMetaClass):
    NA = "NA"
    OPENAI = "OpenAI"
    HUGGING_FACE = "Hugging Face"
    COHERE = "Cohere"


class TransformType(EnumMetaClass):
    RecursiveTransform = "Recursive Text Splitter"
    CharacterTransform = "Character Text Splitter"
    SpacyTransform = "Spacy Text Splitter"
    NLTKTransform = "NLTK Text Splitter"


class IndexerType(EnumMetaClass):
    FAISS = "FAISS"
    CHROMA = "Chroma"
    ELASTICSEARCH = "Elastic Search"


class BotType(EnumMetaClass):
    qna = "Question Answering Bot ❓"
    conversational = "Chatbot 🤖"
