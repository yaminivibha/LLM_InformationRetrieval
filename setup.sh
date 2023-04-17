# Install all dependencies for the project
sudo apt update
sudo apt-get update
sudo apt install python3-pip
pip3 install -U pip setuptools wheel
pip3 install beautifulsoup4 
pip3 install google-api-python-client
pip3 install openAI
pip3 install prettytable
pip3 install pytorch-pretrained-bert
pip3 install -U spacy
pip3 install torch

python3 -m spacy download en_core_web_lg
python3 -m spacy download en_core_web_sm

git clone https://github.com/zackhuiiiii/SpanBERT/
cd SpanBERT
# TODO: make sure this works
pip3 install -r requirements.txt
bash download_finetuned.sh

# move back to repo's root dir
cd ..
mv lib ./SpanBERT/lib

# Moving files into correct dir structure for running:
# File directory setup. These files need to be in the
# same directory as the SpanBERT directory.
mv GPT3Extractor.py ./SpanBERT
mv main.py ./SpanBERT
mv QueryExecutor.py ./SpanBERT
mv SpanBertExtractor.py ./SpanBERT

echo "all set up! :)"