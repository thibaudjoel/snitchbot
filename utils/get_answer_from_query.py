import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI


def get_answer(question, topic):
    df = pd.read_csv("./data.csv")

    agent = create_pandas_dataframe_agent(
        ChatGoogleGenerativeAI(
            temperature=0,
            model="gemini-1.5-pro",
        ),
        df,
        verbose=True,
        allow_dangerous_code=True,
    )
    response = agent.invoke(
        f"Meine Frage ist: {question} \n Sie betrifft das Abstimmungsthema: {topic}"
    )
    return response
