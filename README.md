# Information Extraction from Natural Language on the Web using LLMs and Iterative Set Expansion

**Yamini Ananth, Erin Liang** 

## About

Implementation of an information extraction system that extracts structured information that is embedded in the natural language on webpages. Project uses the Google Custom Search API for the actual retrieval of results. 

This project implements two approaches to extract information (relations) from the web. The desired approach can be specified in the command line.

1. SpanBERT
2. GPT-3 API

Currently four types of relations are supported: **Schools_Attended, Work_for, Live_in,** and **Top_Member_Employees**.

# File Structure

```markdown
├── llm_ise
│   ├── lib
│   │   └── utils.py
├── main.py
├── EntityExtractor.py
├── QueryExecutor.py
└── SpanBertExtractor.py
├── README.md <-- You're here now!
└── setup.sh
```

| Filename                       | Description                                                                                     
|--------------------------------|----------------------------------------------------------------------------------------------------|
| `setup.sh`                     | Bash script for setting up environment                                                             |   
| `GPT3Extractor.py`             | Creates objects that process text using spaCy and extract using GPT3                               |
| `SpanBertExtractor.py`         | Creates objects that process text using spaCy and extract using spanBERT                           |
| `QueryExecutor.py`             | Creates class for query execution, response handling, and input processing                         |      
| `main.py`                      | Main function that handles the control flow                                                        | 
| `utils.py`                     | Utilities for processing documents + urls                                                          |
| `spacy_help_functions.py`      | Utilities for processing documents w/ spaCy                                                        |
|                                | sourced from [here](http://www.cs.columbia.edu/~gravano/cs6111/Proj2/spacy_help_functions.py)      |


# How To Run

<aside>
🍞 All commands necessary to install the required software, dependencies, and run the program.  Note: this code will *not work* locally on M1 Macs because of limitations with the spaCy implementation we are using! Using a VM with Ubuntu is your best bet here. 

</aside>

### Installing Dependencies

- Note: It is advised that you run the setup scripts in a virtual environment to manage your python library versions. For creating and activating virtual environments with the OS we used on VM instances in developing this project (Ubuntu 18.04 LTS), see [this guide](https://linuxize.com/post/how-to-create-python-virtual-environments-on-ubuntu-18-04/). Please start with a completely fresh environment. 

Clone and navigate to the repository:

```bash
git clone https://github.com/yaminivibha/llm_ise.git
```

```bash
cd <your/path/to/llm_ise>
```

Make sure the setup script is executable by changing the file permissions:

```bash
chmod +x setup.sh
```

From the top level repository, run the setup script:

```bash
bash setup.sh
```

- This setup script will install all the requirements and also create the correct file directory structure* for running the program. 
    - **We need to move our scripts around because the main file must be in the same folder as the SpanBERT and helper functions. There are a number of relative paths inside the SpanBERT module that will fail otherwise.*
- The script creates the following directory structure:

```markdown
├── proj2
│   └── SpanBERT
│		  ├── lib
│         │    └── utils.py
│         ├── project2.py    
|  	      ├── EntityExtractor.py
│         ├── QueryExecutor.py
|         └── SpanBertExtractor.py
├── README.md <-- You're here now!
└── setup.sh
```

### Running The Program

Make sure you are in the base repository (which should be the case if following the library installation instructions)

```markdown
$ pwd
<your/path/to/proj2>
```

Then run the project with:

```bash
usage: SpanBERT/main.py [-h] (-spanbert | -gpt3)
                   custom_search_key google_engine_id openai_secret_key r t q
                   k
```

- For our `Google Custom Search Engine JSON API Key` and `Google Engine ID` to run the project, see [Credentials section](https://www.notion.so/ReadMe-2aaf81e050e246ddbb4a69246850c768).

Example commands with the two different types of annotators (`-spanbert` and `-gpt3`)

- extract at least 5 relations of the form Schools_Attended with minimum confidence of 0.7, using spanBERT to annotate the text. “mark zuckerberg harvard” is given as an example tuple that satisfies the desired relation.
    - openAI key is ignored since we are using spanBERT

```bash
python3 SpanBERT/main.py -spanbert AIzaSyDQTz-AzhWHv-Qbk3ADyPG4hFb3Z6PkLHM  45add40315937647f 00000 1 0.7 "mark zuckerberg harvard" 5 
```

- extract at least 35 relations of the form **Work_For**, using GPT3 to annotate the web text. “sundar pichai google” is given as an example tuple that satisfies the desired relation.
    - confidence value is ignored because the gpt3 model is used.

```bash
python3 main.py -gpt3 \
AIzaSyA2-F4UJII_nMxcwkFAY3232hIztCCnJ5U  \
02f24d49c72384af0 <openai_secret_key> \
2 0.7 "sundar pichai google" 35
```

## Parameters

| Parameter | Meaning | Context |
| --- | --- | --- |
| -gpt3 or -spanbert | model | SpanBERT or GPT-3. Exactly one of these two flag must be raised. |
| r  | relation | integer between 1 and 4
• 1 is for Schools_Attended
• 2 is for Work_For
• 3 is for Live_In
• 4 is for Top_Member_Employees |
| t | extraction confidence threshold | float between (0,1)
 which is the minimum extraction confidence that we request for the tuples in the output; t is ignored if we are using -gpt3 |
| q | seed query  | list of words in double quotes corresponding to a plausible tuple for the relation to extract (e.g., "bill gates microsoft" for relation Work_For) |
| k | num requested tuples | integer greater than 0;
number of tuples that we request in the output |

# Internal Design Description

## External Libraries:

| Library | Usage/Reason for Use |
| --- | --- |
| argparse | Handling complex command line arguments |
| BeautifulSoup | Web scraping based on URL |
| spaCy | Processing text and extracting initial relations |
| OpenAI API | Connecting to GPT-3,  text-davinci-003 model, for LLM based NER |
| SpanBERT | As implemented by Zach Hui [here](https://github.com/zackhuiiiii/SpanBERT), used functions involved with SpanBERT prediction for pretrained Transformer-based NER |
| prettytable | Generating final tables in output/transcripts |

## High-level components

| Class | Role |
| --- | --- |
| QueryExecutor | Handles user arguments for given queries; constructs new queries; evaluates iteration continuation criteria; maintains list of seen tuples & seen queries; processes text from URLs |
| GPT3Extractor | Takes processed text; evaluates sentence by sentence with spaCy for existence of valid subject/object pairs; runs GPT3 one-shot entity extraction; returns set of extracted entities for a given document.  |
| SpanBertExtractor | Takes processed text; goes sentence by sentence with spaCy and generating valid subject/object pairs; runs SpanBERT prediction/confidence evaluation; returns set of extracted entities + confidences for a given document.  |

# Program Control Flow

1. In `main.py` , user-inputted arguments are parsed and used to initialize a QueryExecutor object. Depending on which switch is called (`-gpt3` vs `-spanbert` ) the appropriate `Extractor` is created (`GPT3Extractor` or `SpanBertExtractor`). 
2. For the first iteration, the top 10 results are generated using the seed query. For each of the top 10 results, plain text and entities are extracted as described in detail below. 
3. If *k* valid relations are extracted, then the program terminates, printing a table of all extracted relations. Else, it goes onto another iteration using a newly generated query (as described below for each respective `Extractor`) to find and parse 10 more results . 
4. In the case where *k* tuples have not been found, but all possible queries have been exhausted, the program terminates gracefully. 

## Extracting Plain Text From Web Page

- Get the full HTML of a webpage using `requests.get`, setting a max timeout limit of 5 seconds.
- Pass the URL to a `BeautifulSoup` object for processing.
- Find all `<p>` blocks and extract the text. Given that the goal of the pipeline is to extract entity relations from sentences, excluding headers and section titles would have minimal impact. However, we can consider exploring the [impact of including these in future work.](#future-work-👋)
- Truncate the text to its first 10,000 characters (for efficiency) and discard the rest.
- On the truncated text, remove all whitespace and trailing characters as outlined by Zheng Hui [here](https://edstem.org/us/courses/34785/discussion/2831362).
- If a URL times out or has a processing error, move on to the next URL (even if it means processing < 10 URLs in one iteration).

## Extracting Entities Using spaCy

- For a given document of text, after being pre-processed, we follow a different entity relation extraction process for SpanBERT and for GPT-3
- For SpanBERT, we largely follow the NER extraction process as outlined by [example relation extraction code](http://www.cs.columbia.edu/~gravano/cs6111/Proj2/#:~:text=example_relations.py) and filter out the entities based on the target entities of interest that were given in the user’s command line arguments.
- Because extracting relations is expensive downstream, we verify that named entity pairs extracted by spaCy have the correct entity types for the relation before passing them on (for example: “Work_For” requires a PERSON as a subject and an ORGANIZATION as an object).

## SpanBERT (and SpanBertExtractor)

- uses the sentences and named entity pairs extracted by spaCy as input to **SpanBERT** to predict the corresponding relations.
- After spanBERT prediction, we identify the tuples that have an associated extraction confidence of at least **t** and add them to set **X** (maintained in SpanBertExtractor object as instance variable `relations` ).
- When the same tuple is extracted multiple times, we maintain the highest confidence across extractions.

```markdown
Subject: Zuckerberg	Object: Y Combinator's Startup School	Relation: no_relation	Confidence: 1.00
Subject: Zuckerberg	Object: Stanford University	Relation: no_relation	Confidence: 0.76
Subject: Zuckerberg	Object: CFO	Relation: no_relation	Confidence: 1.00
Subject: Facebook.	Object: CFO	Relation: no_relation	Confidence: 1.00
Subject: Zuckerberg	Object: Facebook	Relation: no_relation	Confidence: 1.00
Subject: Zuckerberg	Object: MIT Technology Review's	Relation: no_relation	Confidence: 0.99
Subject: Zuckerberg	Object: 35.[46]	Relation: no_relation	Confidence: 1.00
Subject: Zuckerberg	Object: Vanity Fair	Relation: no_relation	Confidence: 1.00
```

### Improving the quality of the tuples extracted from SpanBERT: Trade-offs between quality and num of iterations

- We noticed that some of the tuples extracted had the label `no_relation` . More often than not, the tuples extracted would be wrong for the target relation.
- We tried to add one additional constraint: that the spaCy extracted *relation* either be ‘no_relation’ or the desired output relation.
    - Generally speaking, excluding ‘no_relation’ meant we needed to run through *far more iterations* (up to 3x) than otherwise, and not all ‘no_relation’ outputs were low quality.
- output for when we restrict SpanBERT’s predicted relation type:

```markdown
================== ALL RELATIONS for Schools_Attended ( 10 ) =================
+------------+---------------------+-------------------------+
| Confidence |       Subject       |          Object         |
+------------+---------------------+-------------------------+
| 0.9862302  |      Zuckerberg     |    Harvard University   |
| 0.9823656  | Norman R. Augustine |    Harvard University   |
| 0.9816164  |   Dustin Moskovitz  |    Harvard University   |
| 0.97826606 |   Andrew McCollum   |    Harvard University   |
| 0.97403777 |   Eduardo Saverin   |    Harvard University   |
| 0.9737499  |      Zuckerberg     | Phillips Exeter Academy |
| 0.96366817 |     Chris Hughes    |    Harvard University   |
| 0.95621806 |    Priscilla Chan   |     Harvard College     |
|  0.925353  |   Mark Zuckerberg   |    Harvard University   |
| 0.7308001  |      Zuckerberg     |         Harvard         |
+------------+---------------------+-------------------------+
```

- output when there are no restrictions (as you can see the quality is significantly decreased — but it terminates with only one iteration!)

```markdown
+------------+-------------------------------+------------------------------------------------+
| Confidence |            Subject            |                     Object                     |
+------------+-------------------------------+------------------------------------------------+
| 0.9998808  |           Zuckerberg          |                    Facebook                    |
| 0.99973303 |           Zuckerberg          |                    Harvard                     |
| 0.9996634  |           Zuckerberg          |         Y Combinator's Startup School          |
| 0.99963194 |           Moskovitz           |                   Dartmouth                    |
| 0.9996281  |      Yale.[36] Zuckerberg     |                    Stanford                    |
| 0.99957097 |           Zuckerberg          |                   Face Books                   |
| 0.99956334 |           Zuckerberg          |                    Stanford                    |
| 0.9995452  |           Moskovitz           |                    Stanford                    |
| 0.9994899  |           breach.[8]          |                 Transportation                 |
| 0.99947655 | shareholder.[1][2] Zuckerberg |               Harvard University               |
| 0.99941176 |           Zuckerberg          |                      CFO                       |
| 0.9988934  |      Yale.[36] Zuckerberg     |                   Dartmouth                    |
| 0.9987791  |           Zuckerberg          |            the Synapse Media Player            |
| 0.9987086  |           Arie Hasit          |                   Face Books                   |
| 0.99863243 |           Zuckerberg          |                   New Yorker                   |
| 0.9985324  |        Tyler Winklevoss       |             HarvardConnection.com              |
| 0.9985181  |         Divya Narendra        |             HarvardConnection.com              |
| 0.99820966 |           Zuckerberg          |                   Dartmouth                    |
| 0.9980716  |           Moskovitz           |       University of Pennsylvania, Brown        |
| 0.99805737 |             Karen             |                    Kempner                     |
| 0.99793935 |          Facebook.[42         |                      CFO                       |
| 0.99788886 |           Zuckerberg          |                 The New Yorker                 |
| 0.99775374 |       Cameron Winklevoss      |             HarvardConnection.com              |
```

## GPT-3 Based NER Extraction

- We use the LLM GPT-3 for named entity extraction in a one-shot learning case.
- We use spaCy to extract entities from sentences in text we extract from the internet. For a given sentence, we check if any pairs of entities produce an appropriate subject-object pair. If so, we pass that sentence (untagged) to GPT3 for entity extraction.

## GPT-3 Prompting

- We specifically format our prompt using JSON blob formatting in the output, with all the relevant quotation marks and punctuation marks. We provide one of the following four output examples depending on the relation we hope to extract:

```jsx
1: '{"PERSON": "Jeff Bezos", "RELATION": "Schools_Attended", "ORGANIZATION": "Princeton University"}',
2: '{"PERSON": "Alec Radford", "RELATION": "Work_For", "ORGANIZATION":"OpenAI"}',
3: '{"PERSON":"Mariah Carey", "RELATION":"Live_In", LOCATION:"New York City"}',
4: '{"ORGANIZATION": "Nvidia","RELATION":"Top_Member_Employees", "PERSON": "Jensen Huang"}',
```

- In our prompt, we also provide example sentences from which the relations can be extracted. They are as follows.

```jsx
		1: "Jeff Bezos graduated from Princeton University.",
    2: "Alec Radford has recently announced he will switch employers to OpenAI.",
    3: "Mariah Carey has a home in Manhattan, New York City.",
    4: "Jensen Huang is the CEO of Nvidia.",
```

- We further provide emphasis/aids in our prompts for each specific relation

```jsx
		1: "Ensure ORGANIZATION is a School, like a University or College.",
    2: "Ensure ORGANIZATION is a Company.",
    3: "Ensure LOCATION is a real world location - like a City, State, or Country.",
    4: "Ensure ORGANIZATION is a Company which has employed PERSON",
```

- We attempted to add more natural language in our prompts (see below)
    - but this increased the amount of iterations the program took to find **k** tuples to extract, even though the tuples that it did extract were more accurate. Because terminating with a lesser number of iterations is the primary goal, we decided to stick with the previous prompt.

```bash
"Extract the name of a school where a person attended. 
Output in the following format: 
[PERSON:PERSON, RELATION:Schools_Attended, ORGANIZATION:SCHOOL].
Ensure ORGANIZATION is a School, like a University or College."
```

## Parsing GPT3 Outputs

- To process the tuples that have been extracted from GPT-3, we convert them to dictionaries using `json.dumps`. If we are unable to handle the object returned by GPT-3 as a JSON blob, we simply move on.
- Next, we have to handle bad subject/object outputs from GPT-3.

For instance, in this sentence, although spaCy found that there was a valid sub/obj pairing, GPT-3 cannot find one and thus returns an empty object. 

```markdown
Prompt: In a given sentence, find relations where PERSON Schools_Attended SCHOOL. 
Output the following: {"PERSON":"PERSON", "RELATION":"Schools_Attended", "ORGANIZATION":"SCHOOL"}. 
Ensure ORGANIZATION is a School.

Example Input: 'Jeff Bezos is an alumnus of Princeton University.' 
Example Output: {"PERSON": "Jeff Bezos", "RELATION": "Schools_Attended", "ORGANIZATION": "Princeton University"}.

Input: Gates has an older sister Kristi (Kristianne) and a younger sister Libby. Output:
GPT-3 Predicted Relation:  {"PERSON": "Gates", "RELATION": "Schools_Attended", "ORGANIZATION": ""}
Relation:  {"PERSON": "Gates", "RELATION": "Schools_Attended", "ORGANIZATION": ""}
Error parsing GPT output: 'NoneType' object is not subscriptable
```

To handle this, a set of rules are necessary to remove invalid tuples generated by GPT3. If any of the following situations occur, we simply move on to the next sentence. 

- GPT-3-generated object has invalid keys or string formatting
- Subject/Object are empty strings, “N/A”, “None”
- Relation does not *exactly* match the relation we are seeking (example: “RELATION”: “Works_For” ✅; “RELATION”: “Works For” ❌)
- Subject *is* or *contains* a pronoun (example: “PERSON”: “He” ❌). We are looking to find NERs on real life individuals and pronouns are too non-specific/don’t align with our goals.
- Subject contains a conjunction (example: “PERSON”: “Bill and Melinda” ❌). The goal is to find atomic NERs, which compound subjects violate.

### Experimenting with prompts

- Although fun/entertaining, prompt engineering is expensive per token, so we only did a few trials for each change we made to our prompt.
- Adding emphasis along the lines of “Ensure that ORGANIZATION is a School.” reduced the amount of erroneous OBJECTS identified slightly (tradeoff here between costs and benefits is marginal.
    - For example, before adding “ensure…school”: 4/10 results are not schools

```jsx
+-----------------+-------------------------------------------------------------------------+
| Subject         | Object                                                                  |
+-----------------+-------------------------------------------------------------------------+
| Mark Zuckerberg | Phillips Exeter Academy                                                 |
| Zuckerberg      | Harvard                                                                 |
| Mark Zuckerberg | Harvard University                                                      |
| Mark Zuckerberg | Netscape CFO Peter Currie                                                |
| Zuckerberg      | New Yorker                                                              |
| Mark Zuckerberg | Stanford University                                                     |
| Zuckerberg      | MIT Technology Review                                                   |
| Zuckerberg      | United States Senate Committee on Commerce, Science, and Transportation |
| Zuckerberg      | High School                                                             |
| Zuckerberg      | Forbes                                                                  |
+-----------------+-------------------------------------------------------------------------+
```

- After: (3 not real schools - 10% improvement)

```jsx
+-----------------+------------------------------+
| Subject         | Object                       |
+-----------------+------------------------------+
| John F Kennedy  | NASA space center            |
| Mark Zuckerberg | Phillips Exeter Academy      |
| Zuckerberg      | Phillips Exeter Academy      |
| Zuckerberg      | Harvard                      |
| Mark Zuckerberg | Stanford University          |
| Zuckerberg      | MIT Technology Review        |
| Mark Zuckerberg | Harvard                      |
| Mark Zuckerberg | White Plains, New York, U.S. |
| Mark Zuckerberg | Harvard University           |
| Zuckerberg      | High School                  |
+-----------------+------------------------------+
```

# Credentials for Testing

- **Google Custom Search Engine JSON API Key:** AIzaSyA2-F4UJII_nMxcwkFAY3232hIztCCnJ5U
- **Engine ID**: 02f24d49c72384af0
- Please use your own OpenAI secret key!

# Future Work 👋

- Parsing more HTML tags other than <p>
    - We also considered parsing other HTML tags, specifically headers. However, we decided against this because
        1. The header and section header tags don’t often contain complete sentences and the goal of this pipeline is to extract entities from full sentences
        2. The header and section header tags often contain summaries of what is enumerated in the subsequent <p> tags, so running the relation extraction system on these additional sentence fragments would likely yield in redundant relations.
- further prompt engineering and quantifying tradeoffs between price vs number of iterations vs output quality. GPT-3 is expensive! SpanBERT is also expensive!
