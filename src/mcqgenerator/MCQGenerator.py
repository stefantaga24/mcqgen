import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.logger import logging

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback

load_dotenv()

key = os.getenv("OPENAI_API_KEY2")

llm = ChatOpenAI(openai_api_key = "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", model_name = "gpt-3.5-turbo", temperature = 0.7)

TEMPLATE3 = """
Text:{text}

You are an amazing text analyser. Given the above text, it is your job
to extract the main subject of the presentation. Make it as consice as possible and it should not be more than 2 words.
"""

TEMPLATE ="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job
to create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be related to the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide.
Ensure to make {number} MCQs


### RESPONSE_JSON
{response_json}

"""

TEMPLATE2 ="""

You are an expert english grammarian and write. Given a Multiple Choice Quiz  for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for the complexity
if the quiz is not at par with the cognitive and analytical abilities of the students,\
update the quiz questions which need to be changed and change the tone such that it perfectly fits the students.
QUIZ_MCQs:
{quiz}

Check from an expert English writer of the above quiz:
"""
quiz_subject_prompt = PromptTemplate(
    input_variables=["text"],
    template = TEMPLATE3
)
quiz_subject_chain = LLMChain(llm = llm, prompt = quiz_subject_prompt, output_key ="subject", verbose = True)

quiz_generation_prompt = PromptTemplate(
    input_variables = ["text", "number", "subject", "tone", "response_json"],
    template = TEMPLATE
)

quiz_chain = LLMChain(llm = llm, prompt = quiz_generation_prompt, output_key="quiz", verbose = True)

quiz_evaluation_prompt = PromptTemplate(input_variables=["subject", "quiz"], template = TEMPLATE2)

review_chain = LLMChain(llm = llm , prompt = quiz_evaluation_prompt, output_key= "review", verbose = True)

generate_evaluate_chain =  SequentialChain(chains = [quiz_chain,review_chain] , input_variables=["text","number","subject","tone","response_json"],
                                           output_variables=["quiz","review"], verbose= True)
