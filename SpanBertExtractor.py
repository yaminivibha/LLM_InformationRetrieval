"Defining GPT3 and SpanBert Extractor classes"
import json
from typing import List, Tuple

import openai
import spacy

from lib.spacy_helper_functions import create_entity_pairs, get_entities
from lib.SpanBERT.spanbert import SpanBERT
from lib.utils import ENTITIES_OF_INTEREST, SUBJ_OBJ_REQUIRED_ENTITIES

# spacy.cli.download("en_core_web_sm")


class SpaCyExtractor:
    "Creates a spaCyExtractor object"

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
        parameters:
            doc: the document to extract candidate pairs from
        returns:
            candidate_entity_pairs: a list of candidate entity pairs, where each pair is a dictionary
                                    with the following keys:
                                        - tokens: the tokens in the sentence
                                        - subj: the subject entity
                                        - obj: the object entity
                                        - sentence: the sentence
        """
        candidate_entity_pairs = []
        print(ENTITIES_OF_INTEREST[self.r])
        for i, sentence in enumerate(doc.sents):
            if i % 5 and i != 0:
                print("        Processed {i} / {num_sents} sentences")
            # print("Processing sentence: {}".format(sentence))
            # print("Tokenized sentence: {}".format([token.text for token in sentence]))
            ents = get_entities(sentence, ENTITIES_OF_INTEREST[self.r])
            # This prints all the entities that spaCy extracts from the sentence.
            # print("spaCy extracted entities: {}".format(ents))

            # Create entity pairs.
            sentence_entity_pairs = create_entity_pairs(
                sentence, ENTITIES_OF_INTEREST[self.r]
            )
            # Filter as we go.
            candidates = self.filter_candidate_pairs(sentence_entity_pairs)
            for candidate in candidates:
                candidate["sentence"] = str(sentence)
                candidate_entity_pairs.append(candidate)

                print("                === Extracted Relation ===")
                print(f"                Sentence:  {sentence}")
                print(
                    f"                Subject: {candidate['subj'][0]} ; Object: {candidate['obj'][0]} ;"
                )
                print("                Adding to set of extracted relations")
                print("                 ==========")

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

        # This info, formatted, should be printed in extract_candidate_pairs.
        # print("Filtered target_candidate_paris: {}".format(target_candidate_pairs))
        return target_candidate_pairs


class spanBertPredictor(SpaCyExtractor):
    def extract_candidate_pairs(self, doc) -> List[Tuple[str, str]]:
        """
        Extract candidate pairs from a given document using spaCy
        parameters:
            doc: the document to extract candidate pairs from
        returns:
            candidate_entity_pairs: a list of candidate entity pairs, where each pair is a dictionary
                                    with the following keys:
                                        - tokens: the tokens in the sentence
                                        - subj: the subject entity
                                        - obj: the object entity
                                        - sentence: the sentence
        """
        candidate_entity_pairs = []
        print(ENTITIES_OF_INTEREST[self.r])
        for i, sentence in enumerate(doc.sents):
            if i % 5 and i != 0:
                print("        Processed {i} / {num_sents} sentences")
            # print("Processing sentence: {}".format(sentence))
            # print("Tokenized sentence: {}".format([token.text for token in sentence]))
            ents = get_entities(sentence, ENTITIES_OF_INTEREST[self.r])
            # This prints all the entities that spaCy extracts from the sentence.
            # print("spaCy extracted entities: {}".format(ents))

            # Create entity pairs.
            sentence_entity_pairs = create_entity_pairs(
                sentence, ENTITIES_OF_INTEREST[self.r]
            )
            # Filter as we go.
            candidates = self.filter_candidate_pairs(sentence_entity_pairs)
            for candidate in candidates:
                candidate["sentence"] = str(sentence)
                candidate_entity_pairs.append(candidate)

        return candidate_entity_pairs

    def get_relations(self, text: str) -> List[Tuple[str, str]]:
        """
        Exposed function to take in text and return named entities
        Parameters:
            text: the text to extract entities from
        Returns:
            entities: a list of tuples of the form (subject, object)
        """
        doc = self.nlp(text)
        print("        Annotating the webpage using spacy...")
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
