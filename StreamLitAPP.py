import os
import json
import traceback
import pandas as pd

from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.logger import logging

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
import streamlit as st

from langchain.callbacks import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.MCQGenerator import quiz_subject_chain

from src.mcqgenerator.logger import logging

with open('Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

st.title("MCQs Creator Application with LangChain")

with st.form("user_inputs"):
    #File Upload
    uploaded_file = st.file_uploader("Upload a PDF or text file")

    mcq_count = st.number_input("No. of MCQs", min_value = 3 ,max_value = 50)
    tone = st.text_input("Complexity Level of Questions", max_chars = 20,placeholder = "Simple")
    button = st.form_submit_button("Create MCQs")

    if button and uploaded_file is not None and mcq_count and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(uploaded_file)
                with get_openai_callback() as cb:
                    response1 = quiz_subject_chain(
                        {
                            "text": text
                        }
                    )
                print(response1.get("subject"))
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number" : mcq_count,
                            "subject" : response1.get("subject"),
                            "tone" : tone,
                            "response_json" : json.dumps(RESPONSE_JSON)
                        }
                    )

            except Exception as e:
                print(f"An exception occurred: {e}")
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception details: {e.args}")
                # You can also print the traceback for more information
                traceback.print_exc()
            
            else:
                if isinstance(response,dict):
                    quiz = response.get("quiz", None)
                    quiz.replace("### RESPONSE_JSON","")
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index+1
                            st.table(df)
                            st.text_area(label = "Review", value = response["review"])
                        else:
                            st.error("Error in the table data")
                else:
                    st.write(response)