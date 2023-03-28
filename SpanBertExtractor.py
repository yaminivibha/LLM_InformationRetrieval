"SpanBertPredictor class"
from typing import Dict, List, Tuple

import spacy
from spacy_help_functions import create_entity_pairs, get_entities
from spanbert import SpanBERT

from lib.utils import (
    ENTITIES_OF_INTEREST,
    SUBJ_OBJ_REQUIRED_ENTITIES,
    TARGET_RELATION_PREDS,
)

# spacy.cli.download("en_core_web_sm")


class spanBertExtractor:
    def __init__(self, r, t, model="en_core_web_sm"):
        """
        Initialize a spaCyExtractor object
        Parameters:
            r: the relation to extract
            model: the spaCy model to use
        Instance Variables:
            nlp: the spaCy model
            total_extracted: the total number of relations extracted
            self.relations: a dictionary of relations and their confidence
                            {(subj, obj): confidence}
        """
        self.nlp = spacy.load(model)
        self.spanbert = SpanBERT("./SpanBERT/pretrained_spanbert")
        self.r = r
        self.t = t
        self.total_extracted = 0
        self.relations = {}

    def extract_candidate_pairs(self, doc):
        """
        Extract candidate pairs from a given document using spaCy
        parameters:
            doc: the document to extract candidate pairs from
        returns:
            self.relations: a list of:
                                        - tokens: the tokens in the sentence
                                        - subj: the subject entity
                                        - obj: the object entity
                                        -
        """
        num_sents = len(list(doc.sents))
        extracted_sentences = 0
        extracted_annotations = 0
        print(
            f"        Extracted {num_sents} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ..."
        )

        for i, sentence in enumerate(doc.sents):
            if i % 5 == 0 and i != 0:
                print(f"        Processed {i} / {num_sents} sentences")
            # print("Processing sentence: {}".format(sentence))
            # print("Tokenized sentence: {}".format([token.text for token in sentence]))
            ents = get_entities(sentence, ENTITIES_OF_INTEREST[self.r])
            # This prints all the entities that spaCy extracts from the sentence.
            # print("spaCy extracted entities: {}".format(ents))

            # Create entity pairs.
            sentence_entity_pairs = create_entity_pairs(
                sentence, ENTITIES_OF_INTEREST[self.r]
            )
            # Filter out entity pairs that don't contain the required entities for the relations on a
            # sentence-by-sentence basis. Keep track of the number of sentences that contain at least
            # one entity pair that contains the required entities, as well as the total number of
            # extracted relations for a given webpage.
            candidates = self.filter_candidate_pairs(sentence_entity_pairs)
            if candidates == []:
                continue

            tokens = candidates[0]["tokens"]
            relation_preds = self.extract_entity_relation_preds(candidates)
            for ex, pred in list(relation_preds):
                rel = (ex["subj"][0], ex["obj"][0])
                self.check_relation_prediction(rel, pred, tokens)
            extracted_sentences += 1
            extracted_annotations += len(relation_preds)

        print(
            f"Extracted annotations for  {extracted_sentences}  out of total  {num_sents}  sentences"
        )
        print(
            f"Relations extracted from this website: {extracted_annotations} (Overall: {len(self.relations)})"
        )
        return extracted_annotations

    def check_relation_prediction(self, rel, pred, tokens):
        """
        Checks if a relation has already been seen.
        If seen, checks if the confidence is higher than the previous one.
        Confidence = max(confidence, previous confidence)

        Also checks if the relation is the target relation.

        Parameters:
            rel: the relation to check
            pred: the prediction confidence of the relation
            tokens: the tokens in the sentence
        Returns:
            None
        """
        # This following section greatly improves the quality of the extracted relations,
        # but we leave it commented out because it increases the number of
        # iterations.
        # Check if the relation is the target relation.
        # If it is, go on and check if there is a duplicate with a higher confidence.
        # if pred[0] not in TARGET_RELATION_PREDS[self.r] or pred[0] == "no_relation":
        #     # print(f"!! not in target relation : {pred[0]})")
        #     return
        if pred[1] < self.t:
            return

        # Check if the relation has already been seen.
        if rel not in self.relations:
            self.relations[rel] = pred[1]
            self.print_relation(rel, pred[1], tokens, duplicate=False)
        else:
            if self.relations[rel] < pred[1]:
                self.relations[rel] = pred[1]
                self.print_relation(rel, pred[1], tokens, duplicate=True, status="<")
            elif self.relations[rel] > pred[1]:
                self.print_relation(rel, pred[1], tokens, duplicate=True, status=">")
            else:
                self.print_relation(rel, pred[1], tokens, duplicate=True, status="=")

        return

    def print_relation(
        self, relation, confidence, tokens, duplicate, status=None
    ) -> None:
        """
        Print relation
        Parameters:
            relation: the relation to print
            confidence: the confidence of the relation
            tokens: the tokens in the sentence
            duplicate: whether the relation is a duplicate
        Returns:
            None
        """
        print("                === Extracted Relation ===")
        print(f"                Input tokens: {tokens}")
        print(
            f"                Output Confidence: {confidence} ; Subject: {relation[0]} ; Object: {relation[1]} ;"
        )
        if duplicate:
            if status == "<":
                print(
                    "                Duplicate with higher confidence than existing record. Updating record."
                )

            elif status == ">":
                print(
                    "                Duplicate with lower confidence than existing record. Ignoring this."
                )
            else:
                print(
                    "                Duplicate with same confidence as existing record. Ignoring this."
                )
        print("                ==========")
        return

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
        return target_candidate_pairs

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
        num_extracted_annotations = self.extract_candidate_pairs(doc)
        if len(self.relations) == 0:
            print("No annotations found...")
        return self.relations

    def extract_entity_relation_preds(
        self, candidate_pairs
    ) -> List[Tuple[Tuple[str, str], str]]:
        """
        Extract entity relations and their confidence values from a given document using Spacy.
        Parameters:
            candidate_pairs: a list of candidate pairs to extract relations from
        Returns:
            zip(candidate_pairs, relation_pairs):
        """
        if len(candidate_pairs) == 0:
            print("No candidate pairs found. Returning empty list.")
            return []

        # get predictions: list of (relation, confidence) pairs
        # example: ('per:employee_of', 0.9832898),
        relation_preds = self.spanbert.predict(candidate_pairs)
        # print(relation_preds)
        return [
            (candidate_pairs[i], relation_preds[i]) for i in range(len(candidate_pairs))
        ]
