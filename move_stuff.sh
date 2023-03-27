# Call this after git pulling 
# When dir structure has SpanBert but things have updated outside

# Note - call from inside llm_ise

rm ./SpanBERT/QueryExecutor.py
rm ./SpanBERT/GPT3Extractor.py
rm ./SpanBERT/SpanBertExtractor.py
rm ./SpanBERT/project2.py


mv QueryExecutor.py ./SpanBERT/QueryExecutor.py
mv GPT3Extractor.py ./SpanBERT/GPT3Extractor.py
mv SpanBertExtractor.py ./SpanBERT/SpanBertExtractor.py
mv project2.py ./SpanBERT/project2.py







