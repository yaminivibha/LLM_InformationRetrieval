import spacy
from lib.SpanBERT.spanbert import SpanBERT
from lib.spacy_helper_functions import get_entities, create_entity_pairs
from lib.utils import (
    ENTITIES_OF_INTEREST,
    RELATIONS,
    SEED_PROMPTS,
    SUBJ_OBJ_REQUIRED_ENTITIES,
)
from typing import List, Tuple

spacy.cli.download("en_core_web_sm")


class Extractor:
    def __init__(self, r, model="en_core_web_sm"):
        self.r = r
        self.nlp = spacy.load(model)

    def process(self, text: str) -> spacy.tokens.doc.Doc:
        """
        Process a given text using Spacy
        """
        doc = self.nlp(text)
        return doc

    def get_relations(self, text: str) -> List[Tuple[str, str]]:
        """
        Exposed function to take in text and return named entities
        """
        doc = self.process(text)
        target_candidate_pairs = self.extract_candidate_pairs(doc)

        return target_candidate_pairs

    def extract_candidate_pairs(self, doc) -> List[Tuple[str, str]]:
        """
        Extract candidate pairs from a given document using Spacy
        """
        entity_pairs = []
        for sentence in doc.sents:
            #     print("\n\nProcessing sentence: {}".format(sentence))
            #     print("Tokenized sentence: {}".format([token.text for token in sentence]))
            ents = get_entities(sentence, ENTITIES_OF_INTEREST[self.r])
            #     print("spaCy extracted entities: {}".format(ents))

            # Create entity pairs.
            candidate_pairs = []
            sentence_entity_pairs = create_entity_pairs(
                sentence, ENTITIES_OF_INTEREST[self.r]
            )
            sentence_entity_pairs.append(entity_pairs)
        return sentence_entity_pairs


class spaCyExtractor(Extractor):
    def __init__(self, r, model="en_core_web_sm"):
        self.nlp = spacy.load(model)
        self.spanbert = SpanBERT("../SpanBERT/pretrained_spanbert")
        self.r = r

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

    #     target_candidate_pairs = [
    #         p for p in candidate_pairs if not p["subj"][1] in SUBJ_OBJ_REQUIRED_ENTITIES[self.r]["SUBJ"] and not p["obj"][1] in SUBJ_OBJ_REQUIRED_ENTITIES[self.r]["OBJ"]
    #     ]

    #     for p in candidate_pairs:
    #         if p["subj"][1] in SUBJ_OBJ_REQUIRED_ENTITIES[self.r]["SUBJ"] and \
    #                 p["obj"][1] in SUBJ_OBJ_REQUIRED_ENTITIES[self.r]["OBJ"]:
    # target_candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
    # target_candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Object, e2=Subject

    # entity_pairs = create_entity_pairs(sentence, ENTITIES_OF_INTEREST)
    # return entity_pairs

    def extract_entities(self, candidate_pairs):
        """
        Extract entities and their conf values from a given document using Spacy.
        """
        relation_preds = self.spanbert.predict(
            candidate_pairs
        )  # get predictions: list of (relation, confidence) pairs

        # Print Extracted Relations
        print("\nExtracted relations:")
        for ex, pred in list(zip(candidate_pairs, relation_preds)):
            print(
                "\tSubject: {}\tObject: {}\tRelation: {}\tConfidence: {:.2f}".format(
                    ex["subj"][0], ex["obj"][0], pred[0], pred[1]
                )
            )
        return relation_preds

        # TODO - should be taken care of: focus on target relations
        # '1':"per:schools_attended"
        # '2':"per:employee_of"
        # '3':"per:cities_of_residence"
        # '4':"org:top_members/employees"


class gpt3Extractor(Extractor):
    def __init__(self, r, model="en_core_web_sm"):
        self.r = r
        self.nlp = spacy.load(model)
