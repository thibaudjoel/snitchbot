from langchain_core.prompts import ChatPromptTemplate


def get_prompt():
    message = """
    Du bist ein hilfreicher AI Chatbot und sollst aus der folgenden Frage zu Abstimmungen im Bundestag das Abstimmungsthema bestimmen. Nenne das Abstimmungshema der Frage in ein bis 5 Wörtern.

    Frage:
    {question}
    """

    prompt = ChatPromptTemplate.from_messages([("human", message)])
    return prompt
