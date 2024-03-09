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

load_dotenv()

key = os.getenv("OPENAI_API_KEY2")

llm = ChatOpenAI(openai_api_key = key, model_name = "gpt-3.5-turbo", temperature = 0.7)



