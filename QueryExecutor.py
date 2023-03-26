"""
Query Executor class and methods
"""
from googleapiclient.discovery import build
import regex as re
from typing import List, Tuple, Dict
import requests
from bs4 import BeautifulSoup
from EntityExtractor import spanBertPredictor, gpt3Predictor

# from nltk.tokenize import word_tokenize

from lib.spacy_helper_functions import get_entities, create_entity_pairs
from lib.utils import ENTITIES_OF_INTEREST, RELATIONS, SEED_PROMPTS
from EntityExtractor import spaCyExtractor

# HTML tags that we want to extract text from.
blocks = ["p", "h1", "h2", "h3", "h4", "h5", "blockquote"]


class QueryExecutor:
    "Creates a QueryExecutor object"

    def __init__(self, args) -> None:
        """
        Initialize a QueryExecutor object
        Instance Variables:
            query: the query string
            r: the relation to extract
            t: the extraction confidence threshold
            k: the number of tuples that we request in the output
            spanbert: whether or not to use SpanBERT
            gpt3: whether or not to use GPT-3
            google_engine_id: the Google Custom Search Engine ID
            openai_secret_key: the OpenAI Secret Key
            engine: the Google Custom Search Engine
        """

        self.q = args.q
        self.r = args.r
        self.t = args.t
        self.k = args.k
        self.spanbert = args.spanbert
        self.gpt3 = args.gpt3
        self.custom_search_key = args.custom_search_key
        self.google_engine_id = args.google_engine_id
        self.openai_secret_key = args.openai_secret_key
        self.engine = build("customsearch", "v1", developerKey=args.custom_search_key)
        self.seen_urls = set()
        self.extractor = (
            gpt3Predictor(r=self.r, openai_key=self.openai_secret_key)
            if self.gpt3
            else spanBertPredictor(r=self.r)
        )

    def printQueryParams(self) -> None:
        """
        Prints the query parameters
        """
        print("Parameters:")
        print(f"Client key      = {self.custom_search_key}")
        print(f"Engine key      = {self.google_engine_id}")
        print(f"OpenAI key      = {self.openai_secret_key}")
        if self.spanbert:
            print(f"Method  = spanbert")
        if self.gpt3:
            print(f"Method  = gpt3")
        print(f"Relation        = {RELATIONS[self.r]}")
        print(f"Threshold       = {self.t}")
        print(f"Query           = {self.q}")
        print(f"# of Tuples     = {self.k}")
        return

    def getQueryResult(self, query: str, k) -> List:
        """
        Get the top 10 results for a given query from Google Custom Search API
        Source: https://github.com/googleapis/google-api-python-client/blob/main/samples/customsearch/main.py
        """

        full_res = (
            self.engine.cse()
            .list(
                q=query,
                cx=self.google_engine_id,
            )
            .execute()
        )

        return full_res["items"][0 : k + 1]

    def processText(self, url: str) -> List[str]:
        """
        Get the tokens from a given URL
        """
        # TODO:
        # If you cannot retrieve the webpage (e.g. because of a timeout),
        # you should skip it and move on to the next one, even if this involves
        # processing fewer than 10 webpages in this iteration.

        # Extract the plain text from the URL using Beautiful Soup.
        # If the resulting plain text is longer than 10,000 characters, truncate it
        # for efficiency and discard the rest.
        # We only want to process the text in the <p> tags.
        try:
            page = requests.get(url)
            print("        Fetching text from url ...")
            soup = BeautifulSoup(page.content, "html.parser")
            html_blocks = soup.find_all("p")
            text = ""
            for block in html_blocks:
                # print(f"block: {block}")
                text += block.get_text()

            if text != "":
                text_len = len(text)
                print(
                    f"        Trimming webpage content from {text_len} to 10000 characters"
                )
                preprocessed_text = (text[:10000]) if text_len > 10000 else text
                print(
                    f"        Webpage length (num characters): {len(preprocessed_text)}"
                )

                # Removing redundant newlines and some whitespace characters.
                preprocessed_text = re.sub("\t+", " ", preprocessed_text)
                preprocessed_text = re.sub("\n+", " ", preprocessed_text)
                preprocessed_text = re.sub(" +", " ", preprocessed_text)
                preprocessed_text = preprocessed_text.replace("\u200b", "")

                return preprocessed_text
            else:
                return None
        except Exception as e:
            print(f"Error processing {url}: {e}")
            raise Exception(f"Error processing {url}")

    # def to_plaintext(html_text: str) -> str:
    #     soup = BeautifulSoup(html_text, "html.parser")
    #     extracted_blocks = _extract_blocks(soup.body)
    #     extracted_blocks_texts = [block.get_text().strip() for block in extracted_blocks]
    #     return "\n".join(extracted_blocks_texts)

    # def _extract_blocks(parent_tag) -> list:
    #     extracted_blocks = []
    #     for tag in parent_tag:
    #         if tag.name in blocks:
    #             extracted_blocks.append(tag)
    #             continue
    #         if isinstance(tag, Tag):
    #             if len(tag.contents) > 0:
    #                 inner_blocks = _extract_blocks(tag)
    #                 if len(inner_blocks) > 0:
    #                     extracted_blocks.extend(inner_blocks)
    #     return extracted_blocks

    def parseResult(self, result: Dict[str, str]) -> List[Tuple[str, str]]:
        """
        Parse the result of a query
        """

        url = result["link"]
        entity_pairs = None
        if url not in self.seen_urls:
            self.seen_urls.add(url)
            text = self.processText(url)
            entities = self.extractor.get_relations(text)
        return entities
