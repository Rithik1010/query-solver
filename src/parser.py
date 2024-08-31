import os
import re
import uuid

import pandas as pd
import requests
import weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    YoutubeLoader,
)
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings, OpenAI

from qdrant_client import QdrantClient, models

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


class ContentClassifier:
    """
    This class identifies the content type of a URL.
    """

    def __init__(self):
        self.youtube_pattern = r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/.*"

    def is_url(self, url):
        """
        Checks if the url is a URL.

        Args:
            url (str): The URL to be checked.

        Returns:
            bool: True if the URL is a URL, False otherwise.
        """
        return url.startswith("http")

    def is_youtube_url(self, url):
        """
        Checks if the url is a YouTube URL using a regular expression.

        Args:
            url (str): The URL to be checked.

        Returns:
            bool: True if the URL is from YouTube, False otherwise.
        """
        return re.match(self.youtube_pattern, url) is not None

    def get_content_type(self, url):
        """
        Checks the content type of the URL.

        Args:
            url (str): The URL to get the content type for.

        Returns:
            str: The content type of the URL (e.g., youtube, text/html, pdf).
        """
        if not self.is_url(url):
            return "not url"

        if self.is_youtube_url(url):
            return "youtube"

        response = requests.head(url)
        return response.headers["content-type"]


class DocumentManager:
    """
    This class handles loading documents, splitting text, and storing them in a vector store.
    """

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    COLLECTION_NAME = "website_dump"

    INDEXED_VECTOR_KEYS = ["id", "content_type", "course_id", "module_id", "topic_id"]

    SOURCE_COL = "Source "

    def __init__(self, **kwargs):
        """
        Initializes the DocumentManager with OpenAI API key and Qdrant connection details.
        """
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.openai = OpenAI(
            temperature=0.2,
            openai_api_key=self.OPENAI_API_KEY,
            model_name="gpt-3.5-turbo-instruct",
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

        self.qdrant_client = QdrantClient(
            url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY, port=None, timeout=10000
        )
        self.doc_store = Qdrant(
            client=self.qdrant_client,
            embeddings=self.embeddings,
            collection_name=self.COLLECTION_NAME,
        )

        self.classifier = ContentClassifier()

        if kwargs.get("force_delete") is True:
            self.create_qdrant_collection()

    def create_qdrant_collection(self):
        """
        Creates a collection in Qdrant with appropriate vector and distance configuration.
        """
        self.qdrant_client.recreate_collection(
            self.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1536, distance=models.Distance.COSINE
            ),
        )

        for field in self.INDEXED_VECTOR_KEYS:
            f = self.qdrant_client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name=field,
                field_schema="keyword",
            )

    def get_loader(self, content_type, source):
        """
        Selects the appropriate document loader based on the content type.

        Args:
            content_type (str): The content type of the document (e.g., youtube, text/html, pdf).
            source (str): The URL or file path of the document.

        Returns:
            DocumentLoader: An instance of the appropriate document loader class.
        """
        if "text/html" in content_type:
            return WebBaseLoader(source)
        elif "pdf" in content_type:
            return PyPDFLoader(source)
        elif "youtube" in content_type:
            return YoutubeLoader.from_youtube_url(source, add_video_info=True)
        else:
            return None

    def get_common_metadata(self, content_type: str, course_name: str, row: pd.Series):
        """
        Returns a dictionary of common metadata for a document.

        Args:
            content_type (str): The content type of the document
            course_name (str): The name of the course.
            row (pd.Series): A row from the dataframe.

        Returns:
            dict: A dictionary of common metadata for a document.
        """

        metas = dict()
        metas["course_id"] = course_name
        metas["content_type"] = content_type

        if isinstance(row, pd.Series):
            unique_id = "{}_{}_{}".format(
                row["Module Id"],
                row["Live Session No."].replace("L", "T").replace("0", ""),
                row["Content"],
            )

            metas["id"] = str(uuid.uuid3(uuid.NAMESPACE_DNS, unique_id))

            if row["Module Id"]:
                metas["module_id"] = row["Module Id"]

            if row["Live Session No."]:
                metas["topic_id"] = (
                    row["Live Session No."].replace("L", "T").replace("0", "")
                )

        return metas

    def load_and_store_documents(
        self,
        data_path: str,
        course_name: str,
        filter_module: str = None,
        dry_run: bool = False,
    ):
        """
        Loads documents from a CSV file, classifies their content type,
        splits text into chunks, and stores them in Qdrant along with metadata.

        Args:
            data_path (str): Path to the CSV file containing document information.
            course_name (str): Name of the course
        """
        df = pd.concat(
            pd.read_excel(data_path, sheet_name=None), ignore_index=True
        ).copy()
        df = df[df["Module Id"].notnull()][
            ["Module Id", "Live Session No.", "Topic", "Content", "Source "]
        ]

        if filter_module:
            df = df[df["Module Id"] == filter_module]

        print(f"Total record count: {df.shape}")

        batch_size = 200  # Adjust this value based on your needs and API limits
        text_batch = []
        meta_batch = []

        for index, row in df.iterrows():
            try:
                content_type = self.classifier.get_content_type(row[self.SOURCE_COL])
                loader = self.get_loader(content_type, row[self.SOURCE_COL])

                if not loader:
                    print(
                        f"Unsupported content type: {content_type} for index: {index}"
                    )
                    continue

                document = loader.load()
                texts = self.text_splitter.split_documents(document)

                # Combine text content with metadata and store in Qdrant
                text_meta_pairs = [
                    (
                        text.page_content,
                        {
                            **text.metadata,
                            **self.get_common_metadata(content_type, course_name, row),
                        },
                    )
                    for text in texts
                ]

                 # Directly unpack and append to batches
                for text, meta in text_meta_pairs:
                    text_batch.append(text)
                    meta_batch.append(meta)

                if len(text_batch) >= batch_size:
                    if dry_run:
                        print("This is just a dry run.")
                        return
                    stored_chunks = self.doc_store.add_texts(text_batch, meta_batch)
                    print(
                        f"Stored {len(stored_chunks)} chunks for index: {index} id: {stored_chunks[0]}"
                    )
                    text_batch = []  # Clear the batch for next iteration
                    meta_batch = []  # Clear the metadata batch for next iteration
                else:
                    print(
                        f"Storing chunks into batch now Size: {len(text_batch)} index {index}"
                    )

            # if len(text_meta_pairs) <= 0:
            #     print(f"No metadata for content index: {index} key: {content_type}")
            #     continue

            # chunks, meta = list(zip(*text_meta_pairs))
            # stored_chunks = self.doc_store.add_texts(chunks, meta)
            # print(
            #     f"Stored {len(stored_chunks)} chunks for index: {index} id: {stored_chunks[0]}"
            # )
            except Exception as e:
                print(f"Error for content index: {index} reason: {str(e)}")
                pass

        if len(text_batch) > 0:
            if dry_run:
                print("This is just a dry run.")
                return
            self.doc_store.add_texts(text_batch, meta_batch)
    
        print("#" * 30)
        print("Chunks loaded")
        print("#" * 30)

    def get_collection(self, ids: str):
        return self.doc_store._client.query.get(self.COLLECTION_NAME, properties=ids)

    def load_urls_document(self, data_path: str):
        df = pd.read_csv(data_path)

        print(f"Total record count: {df.shape}")

        df = df[df["contentType"] == "Blog"]

        print(f"Total filtered record count: {df.shape}")

        batch_size = 200  # Adjust this value based on your needs and API limits
        text_batch = []
        meta_batch = []

        for index, row in df.iterrows():
            try:
                source_url = "https://www.novatr.com/blog/{}".format(row["url"])
                content_type = self.classifier.get_content_type(source_url)
                loader = self.get_loader(content_type, source_url)

                if not loader:
                    print(
                        f"Unsupported content type: {content_type} for index: {index}"
                    )
                    continue

                document = loader.load()
                texts = self.text_splitter.split_documents(document)

                # Combine text content with metadata and store in Qdrant
                text_meta_pairs = [
                    (
                        text.page_content,
                        {
                            **text.metadata,
                            "source": "blog",
                            "general_content": True,
                        },
                    )
                    for text in texts
                ]

                # Directly unpack and append to batches
                for text, meta in text_meta_pairs:
                    text_batch.append(text)
                    meta_batch.append(meta)

                    if len(text_batch) == batch_size:
                        stored_chunks = self.doc_store.add_texts(text_batch, meta_batch)
                        print(
                            f"Stored {len(stored_chunks)} chunks for index: {index} id: {stored_chunks[0]}"
                        )
                        text_batch = []  # Clear the batch for next iteration
                        meta_batch = []  # Clear the metadata batch for next iteration
                    else:
                        print(
                            f"Storing chunks into batch now Size: {len(text_batch)} index {index}"
                        )

                # print(f"Stored {len(chunks)} chunks for index: {meta}")
            except Exception as e:
                print(f"Error for content index: {index} reason: {str(e)}")
                pass

        if len(text_batch) > 0:
            self.doc_store.add_texts(text_batch, meta_batch)

        print("#" * 30)
        print("Chunks loaded")
        print("#" * 30)


file_name = "Copy of MCD- Resources.xlsx"
file_path = "dump/{}".format(file_name)
c = DocumentManager(force_delete=False)
c.load_and_store_documents(file_path, "MCDC", dry_run=True)

# c.load_urls_document(file_path)
# m = c.get_collection("79a09c70-76ac-37f5-8960-13d99050c10c")

# print(m)
