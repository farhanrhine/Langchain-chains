from langchain_ollama import ChatOllama
#from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

#load_dotenv()  # it is used to load the environment variables from the .env file when we are using the API key

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatOllama(model='tinydolphin')

parser = StrOutputParser()

chain = prompt | model | parser # pipe operaters it is used to connect the nodes

result = chain.invoke({'topic':'dubai'})

print(result)

chain.get_graph().print_ascii()