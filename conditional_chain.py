from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOllama(model='tinydolphin')

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# Simplified sentiment classification
def classify_sentiment(text: str) -> str:
    """Simple function to classify sentiment as 'positive' or 'negative'"""
    # Convert to lowercase for case-insensitive matching
    text = text.lower()
    # List of positive words/phrases
    positive_words = ['good', 'great', 'awesome', 'amazing', 'love', 'like', 'excellent', 'wonderful']
    # List of negative words/phrases
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'poor', 'worst']
    
    # Count positive and negative words
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    # Return the sentiment with higher count, default to 'positive' if equal
    return 'positive' if pos_count >= neg_count else 'negative'

# Classifier chain with simple sentiment analysis
classifier_chain = RunnableLambda(lambda x: {"sentiment": classify_sentiment(x['feedback']), "feedback": x['feedback']})

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

# response chain
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x['sentiment'] == 'positive', prompt2 | model | parser),
    (lambda x:x['sentiment'] == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment") # this is not a chain but runnable make its chain
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a bad phone'}))

chain.get_graph().print_ascii()

