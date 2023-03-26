import argparse

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
    1: '[PERSON: "Jeff Bezos, RELATION: "Schools_Attended", ORGANIZATION: "Princeton University"]',
    2: '[PERSION: "Alec Radford", RELATION: "Work_For", ORGANIZATION:"OpenAI"]',
    3: '[PERSON:"Mariah Carey", RELATION:"Live_In", LOCATION:"New York City"]',
    4: '[ORGANIZATION: "Nvidia", "Top_Member_Employees", PERSON: "Jensen Huang"]',
}
SEED_SENTENCES = {
    1: "Jeff Bezos is an alumnus of Princeton University.",
    2: "Alec Radford has recently announced he will switch employers to OpenAI.",
    3: "Mariah Carey has a home in Manhattan, New York City.",
    4: "Jensen Huang is the CEO of Nvidia.",
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
