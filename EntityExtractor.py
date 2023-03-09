import spacy
from lib.SpanBERT.spanbert import SpanBERT
from lib.spacy_helper_functions import get_entities, create_entity_pairs
from lib.utils import ENTITIES_OF_INTEREST, RELATIONS, SEED_PROMPTS
from typing import List, Tuple

spacy.cli.download('en_core_web_sm')

class spaCyExtractor:
    def __init__(self, r, model="en_core_web_sm"):
        self.nlp = spacy.load(model)
        self.spanbert = SpanBERT("../SpanBERT/pretrained_spanbert")
        self.r = r

    def process(self, text: str) -> spacy.tokens.doc.Doc:
        """
        Process a given text using Spacy
        """
        doc = self.nlp(text)
        return doc

    def extract_entities(self, text) -> List[Tuple[str, str]]:
        """
        Extract entities from a given document using Spacy
        """
        doc = self.process(text)

        for sentence in doc.sents:
            print("\n\nProcessing entence: {}".format(sentence))
            print("Tokenized sentence: {}".format([token.text for token in sentence]))
            ents = get_entities(sentence, ENTITIES_OF_INTEREST[self.r])
            print("spaCy extracted entities: {}".format(ents))
            # create entity pairs
            candidate_pairs = []
            sentence_entity_pairs = create_entity_pairs(
                sentence, ENTITIES_OF_INTEREST[self.r]
            )
            # for ep in sentence_entity_pairs:
            # # TODO: keep subject-object pairs of the right type for the target relation (e.g., Person:Organization for the "Work_For" relation)
            # candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
            # candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Object, e2=Subject

            # entity_pairs = create_entity_pairs(sentence, ENTITIES_OF_INTEREST)
            # return entity_pairs
        return sentence_entity_pairs
