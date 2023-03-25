"""Main executor file"""
import os
import sys
import argparse

from lib.utils import rValue, tValue, kValue
from QueryExecutor import QueryExecutor


def main():
    """
    Handles command line arguments and calls the QueryExecutor
    """

    # Taking in command line arguments
    parser = argparse.ArgumentParser(
        description="Information Extraction with ISE and LLMs"
    )
    relation_extraction_method = parser.add_mutually_exclusive_group(required=True)
    relation_extraction_method.add_argument(
        "-spanbert", action="store_true", default=False, help="using spanbert"
    )
    relation_extraction_method.add_argument(
        "-gpt3", action="store_true", default=False, help="using gpt3"
    )
    parser.add_argument(
        "custom_search_key", help="Google Custom Search Engine JSON API Key"
    )
    parser.add_argument("google_engine_id", help="Google Custom Search Engine ID")
    parser.add_argument("openai_secret_key", help="OpenAI Secret Key")
    parser.add_argument("r", type=rValue, help="relation to extract; int in [1,4]")
    parser.add_argument(
        "t", type=tValue, help="extraction confidence threshold; float in [0,1]"
    )
    parser.add_argument(
        "q",
        help="list of words in double quotes corresponding to a plausible tuple for the relation to extract",
    )
    parser.add_argument(
        "k", type=kValue, help="number of tuples that we request in the output; int > 0"
    )

    args = parser.parse_args()

    executor = QueryExecutor(args)
    executor.printQueryParams()
    # TODO: printQueryParams before loading the libraries. Laggy rn.
    print("Loading necessary libraries; This should take a minute or so ...")

    # # Initialize the extracted tuples.
    X = set()

    # # Get the top 10 results for the query
    results = executor.getQueryResult(executor.q, 10)
    print(f"=========== Iteration: 0 - Query: {executor.q} ===========")
    for i, item in enumerate(results):
        print(f"URL ( {i} / 10): {item['link']}")
        print(f"title {i}: {item['title']}")
        print(f"snippet {i}: {item['snippet']}")
        print(f"relation pred pairs for {i}: {executor.parseResult(item)}")
        break
    print()


if __name__ == "__main__":
    main()
