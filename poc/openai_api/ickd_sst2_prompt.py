PROMPT = """
Classify the sentiment of the sentence into two classes: "positive" or "negative".
If the sentence expresses a positive sentiment, use the word "positive" to indicate the sentiment.
If the sentence expresses a negative sentiment, use the word "negative" to indicate the sentiment.
Consider the overall tone and specific words used in the sentence.

Example1)
Sentence: A warm, funny, engaging film.
Sentiment: positive

Example2)
Sentence: Terrible acting and a ridiculous plot.
Sentiment: negative

Example3)
Sentence: Brilliantly crafted and remarkably insightful.
Sentiment: positive

Example4)
Sentence: An utterly unconvincing plot.
Sentiment: negative

Consider the following sentence and classify its sentiment.
You can use the words "positive" or "negative" to indicate the sentiment.
Never use other words except "positive" or "negative".

Sentence: {sentence}
Sentiment: 
""".strip()