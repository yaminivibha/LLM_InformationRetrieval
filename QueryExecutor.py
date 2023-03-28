"""
Query Executor class and methods
"""
# import pprint
import re
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from prettytable import PrettyTable

from GPT3Extractor import gpt3Extractor
from lib.utils import RELATIONS
from SpanBertExtractor import spanBertExtractor

# HTML tags that we want to extract text from.


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
            seen_urls: the set of URLs that we have already seen
            used_queries: the set of queries that we have already used
            extractor: the extractor object (either SpanBERTExtractor or GPT-3Extractor)
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
            else spanBertExtractor(r=self.r, t=self.t)
        )

    def printQueryParams(self) -> None:
        """
        Prints the query parameters
        Parameters:
            None
        Returns:
            None
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

    def processText(self, url: str) -> Optional[str]:
        """
        Get the tokens from a given URL
        If webpage retrieval fails (e.g. because of a timeout), it is skipped (None returned)

        Extracts the plain text from the URL using Beautiful Soup.
        If the resulting plain text is longer than 10,000 characters, it is truncated.
        Only the text in the <p> tags is processed.

        Parameters:
            url (str) - the URL to process
        Returns:
            List[str] - the list of tokens
        """

        try:
            print("        Fetching text from url ...")
            page = requests.get(url, timeout=5)
        except requests.exceptions.Timeout:
            print(f"Error processing {url}: The request timed out. Moving on...")
            return None
        try:
            soup = BeautifulSoup(page.content, "html.parser")
            html_blocks = soup.find_all("p")
            text = ""
            for block in html_blocks:
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
            print(f"Error processing {url}: {e}. Moving on ...")
            return None

    def parseResult(self, result: Dict[str, str]) -> None:
        """
        Parse the result of a query.
        Exposed function for use by main function.
        Parameters:
            result (dict) - one item as returned as the result of a query
        Returns:
            None
        """
        url = result["link"]
        if url not in self.seen_urls:
            self.seen_urls.add(url)
            text = self.processText(url)
            if not text:
                return None
            self.extractor.get_relations(text)
        return

    def checkContinue(self) -> bool:
        """
        Evaluate if we have evaluated at least k tuples, ie continue or halt.
        Parameters: None
        Returns: bool (True if we need to find more relations, else False)
        """
        return len(self.extractor.relations) < self.k

    def getNewQuery(self) -> Optional[str]:
        """
        Creates a new query.
        Select from X a tuple y such that y has not been used for querying yet
        Create a query q from tuple y by concatenating
        the attribute values together.
        If no such y tuple exists, then stop/return None.
        (ISE has "stalled" before retrieving k high-confidence tuples.)

        Parameters:
            None
        Returns:
            query (str) if available; else None
        """
        if self.gpt3:
            # Iterating through extracted tuples
            for relation in list(self.extractor.relations):
                # Constructing query
                if self.gpt3:
                    tmp_query = " ".join(relation)
                # Checking if query has been used
                if tmp_query not in self.used_queries:
                    # Adding query to used queries
                    self.used_queries.add(relation)
                    # Setting new query
                    self.q = tmp_query
                    return self.q
            # No valid query found
            return None

        elif self.spanbert:
            # Sort by tuples by confidence
            rels = sorted(
                self.extractor.relations.items(), key=lambda item: item[1], reverse=True
            )
            # TODO: remove after testing
            # pp = pprint.PrettyPrinter(indent=4)
            # pp.pprint(rels)

            queryNotFound = True
            i = 0
            while queryNotFound:
                # No valid query found
                if i >= len(rels):
                    return None
                subj_obj, _pred = rels[i]
                tmp_query = " ".join(subj_obj)

                # Checking if query has been used
                if tmp_query not in self.used_queries:
                    queryNotFound = False
                    # Adding query to used queries
                    self.used_queries.add(tmp_query)
                    # Setting new query
                    self.q = tmp_query
                    return self.q
                i += 1
        return

    def printRelations(self) -> None:
        """
        Print the results of the query, relations in table format
        If -spanbert, sort by confidence (descending)
        Parameters:
            None
        Returns:
            None
        """
        print(
            f"================== ALL RELATIONS for {RELATIONS[self.r]} ( {len(self.extractor.relations)} ) ================="
        )
        table = PrettyTable()
        table.align = "l"
        if self.gpt3:
            table.field_names = ["Subject", "Object"]
            table.add_rows(self.extractor.relations)
        else:
            table.field_names = ["Confidence", "Subject", "Object"]
            for subj_obj, pred in self.extractor.relations.items():
                table.add_row([pred, subj_obj[0], subj_obj[1]])
            table.sortby = "Confidence"
            table.reversesort = True
        print(table)
        return
