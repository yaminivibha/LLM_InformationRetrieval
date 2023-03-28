# Information Extraction from Natural Language on the Web using LLMs and Iterative Set Expansion

Team members: **Yamini Ananth yva2002, Erin Liang ell2147**

## About

Implementation of an information extraction system that extracts structured information that is embedded in the natural language on webpages. Project uses the [Google Custom Search API](http://www.cs.columbia.edu/~gravano/cs6111/proj1.html#:~:text=Google%20Custom%20Search%20API%20(https%3A//developers.google.com/custom%2Dsearch/)) for the actual retrieval of results. 

This project implements two approaches to extract the information (relations). 

1. SpanBERT
2. GPT-3 API

The desired approach can be specified in the command line.

This project was completed as part of the Spring 2023 version of Columbia University’s Advanced Database Systems course (COMS 6111) taught by Professor Luis Gravano at Columbia University.


# How To Run

- Provide all commands necessary to install the required software and dependencies for your program.

### Installing Dependencies

- Note: It is advised that you run the setup scripts in a virtual environment to manage your python library versions. For creating and activating virtual environments with the specific VM instances for this class (Ubuntu 18.04 LTS), see [this guide](https://linuxize.com/post/how-to-create-python-virtual-environments-on-ubuntu-18-04/).

Navigate to the repository:

```bash
cd <your/path/to/proj2>
```

Make sure the setup script is executable by changing the file permissions:

- NOTE: this might not be necessary if Gradescope preserves the file permissions

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
│				  ├── lib
│         │    └── utils.py
│         ├── project2.py    
|  			  ├── EntityExtractor.py
│         ├── QueryExecutor.py
|         └── SpanBertExtractor.py
├── requirements.txt
├── README.pdf <-- You're here now!
└── setup.sh
```

### Actually Running The Program

Make sure you are in the base repository (which should be the case if following the library installation instructions

```markdown
$ pwd
<your/path/to/proj2>
```

Then run the project with:

```bash
usage: SpanBERT/project2.py [-h] (-spanbert | -gpt3)
                   custom_search_key google_engine_id openai_secret_key r t q
                   k
```

- For our `Google Custom Search Engine JSON API Key` and `Google Engine ID` to run the project, see [Credentials section](https://www.notion.so/ReadMe-2aaf81e050e246ddbb4a69246850c768).

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

Example commands with the two different types of annotators (`-spanbert` and `-gpt3`)

- extract at least 5 relations of the form Schools_Attended with minimum confidence of 0.7, using spanBERT to annotate the text. “mark zuckerberg harvard” is given as an example tuple that satisfies the desired relation.

```bash
python3 SpanBERT/project2.py -spanbert \
AIzaSyDQTz-AzhWHv-Qbk3ADyPG4hFb3Z6PkLHM  \
45add40315937647f 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 \
1 0.7 "mark zuckerberg harvard" 5 
```

- extract at least 35 relations of the form Work_For, using GPT3 to annotate the web text. “sundar pichai google” is given as an example tuple that satisfies the desired relation.
    - confidence value is ignored because the gpt3 model is used.

```bash
python3 project2.py -gpt3 \
AIzaSyA2-F4UJII_nMxcwkFAY3232hIztCCnJ5U  \
02f24d49c72384af0 <openai_secret_key> \
2 0.7 "sundar pichai google" 35
```


