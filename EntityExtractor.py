import spacy
from lib.SpanBERT.spanbert import SpanBERT
from lib.spacy_helper_functions import get_entities, create_entity_pairs
from lib.utils import (
    ENTITIES_OF_INTEREST,
    RELATIONS,
    SEED_PROMPTS,
    SEED_SENTENCES,
    SUBJ_OBJ_REQUIRED_ENTITIES,
)
import openai
from typing import List, Tuple

# spacy.cli.download("en_core_web_sm")


class spaCyExtractor:
    def __init__(self, r, model="en_core_web_sm"):
        """
        Initialize a spaCyExtractor object
        Parameters:
            r: the relation to extract
            model: the spaCy model to use
        """
        self.nlp = spacy.load(model)
        self.spanbert = SpanBERT("./lib/SpanBERT/pretrained_spanbert")
        self.r = r

    def extract_candidate_pairs(self, doc) -> List[Tuple[str, str]]:
        """
        Extract candidate pairs from a given document using spaCy
        """
        candidate_entity_pairs = []
        print(ENTITIES_OF_INTEREST[self.r])
        for sentence in doc.sents:
            print("Processing sentence: {}".format(sentence))
            # print("Tokenized sentence: {}".format([token.text for token in sentence]))
            ents = get_entities(sentence, ENTITIES_OF_INTEREST[self.r])
            print("spaCy extracted entities: {}".format(ents))

            # Create entity pairs.
            sentence_entity_pairs = create_entity_pairs(
                sentence, ENTITIES_OF_INTEREST[self.r]
            )
            # Filter as we go
            candidates = self.filter_candidate_pairs(sentence_entity_pairs)
            for candidate in candidates:
                candidate_entity_pairs.append(candidate)
        return candidate_entity_pairs

    def filter_candidate_pairs(self, sentence_entity_pairs):
        # Create candidate pairs. Filter out subject-object pairs that
        # aren't the right type for the target relation.
        # (e.g. don't include anything that's not Person:Organization for the "Work_For" relation)
        candidate_pairs = []
        target_candidate_pairs = []
        for ep in sentence_entity_pairs:
            candidate_pairs.append(
                {"tokens": ep[0], "subj": ep[1], "obj": ep[2]}
            )  # e1=Subject, e2=Object
            candidate_pairs.append(
                {"tokens": ep[0], "subj": ep[2], "obj": ep[1]}
            )  # e1=Object, e2=Subject

        for p in candidate_pairs:
            if (
                p["subj"][1] in SUBJ_OBJ_REQUIRED_ENTITIES[self.r]["SUBJ"]
                and p["obj"][1] in SUBJ_OBJ_REQUIRED_ENTITIES[self.r]["OBJ"]
            ):
                target_candidate_pairs.append(p)
        print("Filtered target_candidate_paris: {}".format(target_candidate_pairs))
        return target_candidate_pairs


class spanBertPredictor(spaCyExtractor):
    def get_relations(self, text: str) -> List[Tuple[str, str]]:
        """
        Exposed function to take in text and return named entities
        Parameters:
            text: the text to extract entities from
        Returns:
            entities: a list of tuples of the form (subject, object)
        """
        doc = self.nlp(text)
        target_candidate_pairs = self.extract_candidate_pairs(doc)
        if len(target_candidate_pairs) == 0:
            print("No candidate pairs found. Returning empty list.")
            return []
        print("target_candidate_pairs: {}".format(target_candidate_pairs))
        entities = self.extract_entity_relation_preds(target_candidate_pairs)
        return entities

    def extract_entity_relation_preds(self, candidate_pairs):
        """
        Extract entity relations and their confidence values from a given document using Spacy.
        Parameters:
            candidate_pairs: a list of candidate pairs to extract relations from
        Returns:
            relation_preds: a list of tuples of the form (relation, confidence)
        """
        if len(candidate_pairs) == 0:
            print("No candidate pairs found. Returning empty list.")
            return []

        # get predictions: list of (relation, confidence) pairs
        relation_preds = self.spanbert.predict(candidate_pairs)
        # Print Extracted Relations
        print("\nExtracted relations:")
        for ex, pred in list(zip(candidate_pairs, relation_preds)):
            print(
                "\tSubject: {}\tObject: {}\tRelation: {}\tConfidence: {:.2f}".format(
                    ex["subj"][0], ex["obj"][0], pred[0], pred[1]
                )
            )
        return relation_preds


class gpt3Predictor(spaCyExtractor):
    def __init__(self, r, openai_key, model="en_core_web_sm"):
        """
        Initialize a gpt3Predictor object
        Parameters:
            r: the relation to extract
            openai_key: the key to use for the OpenAI API
            model: the spaCy model to use
        """
        self.openai_key = openai_key
        self.nlp = spacy.load(model)
        self.spanbert = SpanBERT("./lib/SpanBERT/pretrained_spanbert")
        self.r = r

    def get_relations(self, text: str) -> List[Tuple[str, str]]:
        """
        Exposed function to take in text and return named entities
        Parameters:
            text: the text to extract entities from
        Returns:
            entities: a list of tuples of the form (subject, object)
        """
        doc = self.nlp(text)
        target_candidate_pairs = self.extract_candidate_pairs(doc)
        if len(target_candidate_pairs) == 0:
            print("No candidate pairs found. Returning empty list.")
            return []
        print("target_candidate_pairs: {}".format(target_candidate_pairs))

        return

    def extract_entity_pairs(self, candidate_pairs):
        """
        Extract entity relations
        Parameters:
            candidate_pairs: a list of candidate pairs to extract relations from
        Returns:
            entities: a list of tuples of the form (subject, object)
        """
        entities = []
        for pair in candidate_pairs:
            prompt = self.construct_prompt(pair)
            print("Prompt: {}".format(prompt))
            relation = self.gpt3_complete(prompt)
            print("Relation: {}".format(relation))
            entities.append(relation)
        return entities

    def gpt3_complete(self, prompt):
        """
        Use GPT-3 to complete a prompt
        Parameters:
            prompt: the prompt to complete
        Returns:
            completion: the completion of the prompt
        """
        openai.api_key = ""
        completion = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=1,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"],
        )
        print("GPT-3 Completion: {}".format(completion))
        return completion["choices"][0]["text"]

    def construct_prompt(self, pair):
        """
        Construct a prompt for GPT-3 to complete.
        Parameters:
            candidate_pairs: a single candidate pairs to extract relations from
        Returns:
            prompt: a string to be passed to GPT-3
        """

        seed = f"Given a sentence input, output the following relations: relation_type {RELATIONS[self.r]}. "
        example = f"Example Input: '{SEED_SENTENCES[self.r]}' Example Output: {SEED_PROMPTS[self.r]}."
        input = f"Input: {pair['tokens'].join(' ')}\n"

        return seed + example + input
