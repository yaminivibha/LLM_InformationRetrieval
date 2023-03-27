"""
S Executor class and methods
"""
from typing import Dict, List, Tuple

import regex as re
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from prettytable import PrettyTable

from GPT3Extractor import gpt3Extractor
from SpanBertExtractor import spanBertPredictor
from lib.utils import RELATIONS


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
        self.used_queries = set([self.q])
        self.extractor = (
            gpt3Extractor(r=self.r, openai_key=self.openai_secret_key)
            if self.gpt3
            else spanBertPredictor(r=self.r)
        )
        if self.gpt3:
            self.seen_relations = set()
        elif self.spanbert:
            self.seen_relations = dict()

    def printQueryParams(self) -> None:
        """
        Prints the query parameters
        """
        print("Parameters:")
        print(f"Client key      = {self.custom_search_key}")
        print(f"Engine key      = {self.google_engine_id}")
        print(f"OpenAI key      = {self.openai_secret_key}")
        print(f"Relation        = {RELATIONS[self.r]}")
        if self.spanbert:
            print("Method          = spanbert")
            print(f"Threshold       = {self.t}")
        if self.gpt3:
            print("Method          = gpt3")
            print("Threshold       = XXX")
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
        # If you cannot retrieve the webpage (e.g. because of a timeout),
        # you should skip it and move on to the next one, even if this involves
        # processing fewer than 10 webpages in this iteration.

        # Extract the plain text from the URL using Beautiful Soup.
        # If the resulting plain text is longer than 10,000 characters, truncate it
        # for efficiency and discard the rest.
        # We only want to process the text in the <p> tags.
        try:
            print("        Fetching text from url ...")
            page = requests.get(url, timeout=5)
        except requests.exceptions.Timeout:
            print(f"Error processing {url}: The request timed out.")
            return None
        try:
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
            return None

    def parseResult(self, result: Dict[str, str]) -> List[Tuple[str, str]]:
        """
        Parse the result of a query
        """
        url = result["link"]
        if url not in self.seen_urls:
            self.seen_urls.add(url)
            text = self.processText(url)
            if not text:
                return None
            entities = self.extractor.get_relations(text)
            for entity in entities:    
                if entity not in self.seen_relations:
                    self.seen_relations.add(entity)
        return self.seen_relations

    def checkContinue(self) -> bool:
        """
        Check if we should continue querying
        """
        return len(self.seen_relations) < self.k

    def getNewQuery(self) -> str:
        """
        Creates a new query.
        Select from X a tuple y such that y has not been used for querying yet
        Create a query q from tuple y by concatenating
        the attribute values together.
        If no such y tuple exists, then stop.
        (ISE has "stalled" before retrieving k high-confidence tuples.)
        """

        # Iterating through extracted tuples
        for relation in list(self.seen_relations):
            # Constructing query
            tmp_query = " ".join(relation)
            # Checking if query has been used
            if tmp_query not in self.used_queries:
                # Adding query to used queries
                self.used_queries.add(relation)
                # Setting new query
                self.q = tmp_query
                return self.q
        return None

    def printRelations(self) -> None:
        """
        Print the results of the query, relations
        """
        print(
            f"================== ALL RELATIONS for {RELATIONS[self.r]} ( {len(self.seen_relations)} ) ================="
        )
        table = PrettyTable()
        table.align = "l"
        if self.gpt3:
            table.field_names = ["Subject", "Object"]
            table.add_rows(self.seen_relations)
        else:
            table.field_names = ["Confidence", "Subject", "Object"]
            for rel in self.seen_relations:
                table.add_row(
                    [f"Confidence:{rel[2]}", f"Subject: {rel[0]}", f"Object:{rel[1]}"]
                )
            

        print(table)
        return
