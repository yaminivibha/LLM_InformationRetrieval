if test -f "QueryExecutor.py"; 
then rm ./SpanBERT/QueryExecutor.py
mv QueryExecutor.py ./SpanBERT/QueryExecutor.py
fi

if test -f "GPT3Extractor.py"; 
then rm ./SpanBERT/GPT3Extractor.py
mv GPT3Extractor.py ./SpanBERT/GPT3Extractor.py
fi

if test -f "SpanBertExtractor.py"; 
then rm ./SpanBERT/SpanBertExtractor.py
mv SpanBertExtractor.py ./SpanBERT/SpanBertExtractor.py
fi

if test -f "project2.py"; 
then rm ./SpanBERT/project2.py  
mv project2.py ./SpanBERT/project2.py
fi
