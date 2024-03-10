import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

from constants import SQL_PROMPT_TEMPLATE, CHAIN_PROMPT_TEMPLATE

load_dotenv()


class ChatbotPipeline:
    def __init__(self, csv_file=None, db_name="dati"):
        self.engine = self.create_db_engine(csv_file, db_name)
        self.db = SQLDatabase(engine=self.engine)
        self.llm = ChatOpenAI()
        self.sql_prompt = ChatPromptTemplate.from_template(SQL_PROMPT_TEMPLATE)
        self.chain_prompt = ChatPromptTemplate.from_template(CHAIN_PROMPT_TEMPLATE)

        self.sql_chain = (
            RunnablePassthrough.assign(schema=self.get_schema)
            | self.sql_prompt
            | self.llm.bind(stop="\nSQL Result:")
            | StrOutputParser()
        )

    @staticmethod
    def create_db_engine(csv_file, name):
        engine = None
        if csv_file is not None:
            df = pd.read_csv(csv_file)
            engine = create_engine(f"sqlite:///{name}.db")
            df.to_sql(name, engine, index=False, if_exists="replace")
        return engine

    def get_schema(self, _):
        return self.db.get_table_info()

    def run_query(self, query):
        return self.db.run(query)


    def run_full_chain(self, query):
        full_chain = (
            RunnablePassthrough.assign(query=self.sql_chain).assign(
                schema=self.get_schema,
                response=lambda vars: self.run_query(vars["query"]),
            )
            | self.chain_prompt
            | self.llm
            | StrOutputParser()
        )
        response = full_chain.invoke({"question": query})
        return response


# # Example usage:
# # Assuming you have a CSV file named 'data.csv' that you want to use as the database.
# pipeline = ChatbotPipeline(csv_file="data.csv")
# response = pipeline.run_full_chain("What is the total sales for product A?")
# print(response)
