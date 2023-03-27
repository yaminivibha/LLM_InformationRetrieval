import spacy
from spanbert import SpanBERT
from lib.spacy_helper_functions import get_entities, create_entity_pairs
from lib.utils import (
    ENTITIES_OF_INTEREST,
    RELATIONS,
    SEED_PROMPTS,
    SEED_SENTENCES,
    SUBJ_OBJ_REQUIRED_ENTITIES,
    PROMPT_AIDS,
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
        Instance Variables:
            nlp: the spaCy model
            total_extracted: the total number of relations extracted
        """
        self.nlp = spacy.load(model)
        self.spanbert = SpanBERT("./SpanBERT/pretrained_spanbert")
        self.r = r
        self.total_extracted = 0

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
        num_sents = len(list(doc.sents))
        num_annotated_sents = 0

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
            if candidates != []:
                num_annotated_sents += 1
            for candidate in candidates:
                candidate["sentence"] = str(sentence)
                candidate_entity_pairs.append(candidate)

                print(f"                === Extracted Relation ===")
                print(f"                Input tokens: {candidate['tokens']}")
                print(
                    f"                Output Confidence: FILLER ; Subject: {candidate['subj'][0]} ; Object: {candidate['obj'][0]} ;"
                )
                # TODO: add subj obj pair to set.
                # Discard if the subj obj pair is a duplicate with a lower confidence value.
                print(f"                Adding to set of extracted relations")
                print(f"                ==========")

        num_relations_extracted = len(candidate_entity_pairs)
        self.total_extracted += num_relations_extracted
        print(
            f"        Relations extracted from this website: {num_relations_extracted} (Overall: {self.total_extracted})"
        )
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
        print(f"        Annotating the webpage using spacy...")
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
