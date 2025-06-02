# RAG-System

## Document selection
I allowed the AI to select the document initially, and liked what it chose, so I stuck with it. 
The document is a wiki page about natural language processing, which I think is a bit on the nose.

## The questions and answers
### Your question: What is the ideal embedding dimensionality for effective FAISS search?
Answer: a Chinese phrasebook, with questions and matching answers

### Your question: How does chunk overlap affect the quality of retrieval in a RAG system?
Answer: decrease in computational power (seeMoore's law) and the gradual lessening of the dominance ofChomskyantheories of linguistics (e.g.transformational grammar), whose theoretical (e.g.transformational grammar), whose theoretical underpinnings discouraged the sort ofcorpus linguisticsthat underlies the machine-learning approach to language processing

### Your question: What types of FAISS indexes are best suited for approximate nearest neighbor search?
Answer: data retrieval,knowledge representationandcomputational linguistics

### Your question: How should the prompt be designed to maximize answer relevance using retrieved chunks?
Answer: Given a collection of rules (e.g., a Chinese phrasebook, with questions and matching answers), the computer emulates natural language understanding (or other NLP tasks) by applying those rules to the data it confronts. Up until the 1980s, most natural language processing systems were based on complex sets of hand-written rules

### Your question: What are common pitfalls when integrating chunking and retrieval in a RAG pipeline?
Answer: the steady increase in computational power (seeMoore's law) and the gradual lessening of the dominance ofChomskyantheories of linguistics (e.g.transformational grammar), whose theoretical (e.g.transformational grammar), whose theoretical underpinnings discouraged the sort ofcorpus linguisticsthat underlies the machine-learning approach to language processing

## Analysis
Its pretty clear that some of these answers are pretty off-topic, especially the first one.
Changing the chunk size and overlap lead to some pretty interesting results, looking at question 1 for example, I changed the chunk size from 500 to 5000, this caused some increased loading time on the program but it generated a very different asnwer as a result:

### Your question: What is the ideal embedding dimensionality for effective FAISS search?
Answer: Semantic properties of words. Intermediate tasks (e.g., part-of-speech tagging and dependency parsing) are not needed anymore. Neural machine translation, based on then-newly inventedsequence-to-sequencetransformations, made obsolete the intermediate steps, such as word alignment, previously necessary forstatistical machine translation. Since 2015,[22]the statistical approach has been replaced by theneural networksapproach, usingsemantic networks[23]andword embeddingsto capture semantic properties of words. Intermediate tasks (e.g., part-of-speech tagging and dependency parsing) are not needed anymore. Neural machine translation, based on then-newly inventedsequence-to-sequencetransformations, made obsolete the intermediate steps, such as word alignment, previously necessary forstatistical machine translation.

Talking with my chosen AI, it recommended empirical tuning to improve the quality of the output from the RAG. It also suggested switching to a larger model like facebook's bart large cnn. 
Most interesting, I think though, it recommended providing a better prompt design before asking the question such as, "You are an expert in Retrieval-Augmented Generation (RAG) systems" and "Explain the answer clearly and concisely."
Finally an idea to post-process the output to detect bad answers was also a decent idea, with flawed answers maybe tuning the chunk size and overlap until a desired output is achieved, though I think the latter could lead to massive compute usage for sophisticated questions.
Overall this was a really interesting project that taught me a lot about how RAG systems operate and the role and significance chunks play in output.
