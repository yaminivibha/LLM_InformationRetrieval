import argparse


PRONOUNS_AND_CONJUNCTIONS = [
    "and",
    "or",
    "nor",
    "but",
    "yet",
    "so",
    "for",
    "and",
    "he",
    "she",
    "him",
    "her",
    "his",
    "hers",
    "himself",
    "herself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "it",
    "its",
    "itself",
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
]
ENTITIES_OF_INTEREST = {
    0: ["PERSON", "ORGANIZATION", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"],
    1: ["PERSON", "ORGANIZATION"],
    2: ["PERSON", "ORGANIZATION"],
    3: ["PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"],
    4: ["ORGANIZATION", "PERSON"],
}

RELATIONS = {
    1: "Schools_Attended",
    2: "Work_For",
    3: "Live_In",
    4: "Top_Member_Employees",
}

SEED_PROMPTS = {
    1: '{"PERSON": "Jeff Bezos", "RELATION": "Schools_Attended", "ORGANIZATION": "Princeton University"}',
    2: '{"PERSON": "Alec Radford", "RELATION": "Work_For", "ORGANIZATION":"OpenAI"}',
    3: '{"PERSON":"Mariah Carey", "RELATION":"Live_In", LOCATION:"New York City"}',
    4: '{"ORGANIZATION": "Nvidia","RELATION":"Top_Member_Employees", "PERSON": "Jensen Huang"}',
}
SEED_SENTENCES = {
    1: "Jeff Bezos graduated from Princeton University.",
    2: "Alec Radford has recently announced he will switch employers to OpenAI.",
    3: "Mariah Carey has a home in Manhattan, New York City.",
    4: "Jensen Huang is the CEO of Nvidia.",
}

PROMPT_AIDS = {
    1: "PERSON Schools_Attended SCHOOL. Output the following: [PERSON:PERSON, RELATION:Schools_Attended, ORGANIZATION:SCHOOL]. Ensure ORGANIZATION is a School, like a University or College.",
    2: "PERSON Work_For COMPANY. Output the following: [PERSON:PERSON, RELATION:Work_For, ORGANIZATION:COMPANY]. Ensure ORGANIZATION is a Company.",
    3: "PERSON Live_In LOCATION. Output the following: [PERSON:PERSON, RELATION:Live_In, LOCATION:LOCATION]. Ensure LOCATION is a real world location - like a City, State, or Country.",
    4: "COMPANY Top_Member_Employees PERSON. Output the following: [ORGANIZATION:COMPANY, RELATION:Top_Member_Employees, PERSON:PERSON]. Ensure ORGANIZATION is a Company.",
}

SUBJ_OBJ_REQUIRED_ENTITIES = {
    1: {"SUBJ": ["PERSON"], "OBJ": ["ORGANIZATION"]},
    2: {"SUBJ": ["PERSON"], "OBJ": ["ORGANIZATION"]},
    3: {
        "SUBJ": ["PERSON"],
        "OBJ": ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"],
    },
    4: {"SUBJ": ["ORGANIZATION"], "OBJ": ["PERSON"]},
}

# Values are SpanBERT's target relation prediction IDs.
# We use these to filter out predictions that are not the target relation.
TARGET_RELATION_PREDS = {
	1: ["per:schools_attended"],
	2: ["per:employee_of"],
	3: ["per:countries_of_residence", "per:cities_of_residence", "per:statesorprovinces_of_residence"],
	4: ["org:top_members_employees"]
}


def tValue(string) -> float:
    value = float(string)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError("t value has to be a float between 0 and 1")
    return value


def rValue(string) -> int:
    value = int(string)
    if value < 1 or value > 4:
        raise argparse.ArgumentTypeError("r value has to be an integer between 1 and 4")
    return value


def kValue(string) -> int:
    value = int(string)
    if value < 1:
        raise argparse.ArgumentTypeError("k value has to be an integer greater than 0")
    return value
