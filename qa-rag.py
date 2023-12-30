"""
The script allows you to interact with your notes stored in Notion using natural language. 
You can ask questions, and the script will retrieve the most relevant answers from your notes. 
It uses a retrieval-augmented model to search through your notes and provide accurate responses. 
Additionally, it provides citations for the sources of information, so you can easily refer back to the original notes.
"""

from operator import itemgetter
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain.memory import ConversationBufferMemory
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Chroma

import dotenv
import os
import streamlit as st

dotenv.load_dotenv()

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

if not os.path.isdir('./chroma'):
    print('Indexing for the first time... may take a while')
    from notion_rag import import_pipeline
    import_pipeline()
    print('Import completed')

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma")
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
}
# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
    "docs": itemgetter("docs"),
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer

def process_inputs(inputs):
    result = final_chain.invoke(inputs)

    answer = result['answer'].content
    citations = set(source.metadata['title'] for source in result['docs'])

    return answer, citations

# if __name__ == '__main__':
#     print('Press CTRL-D to stop.\n')
#     while True:
#         user_input = input("Enter your question: ")

#         inputs = {"question": user_input}
#         answer, citations = process_inputs(inputs)

#         print(f"Answer: {answer}")
#         print("Citations: ")

#         for title in citations:
#             print(f" - {title}")

st.title("🦜🔗 Talk to your Notion notes!")


with st.form("my_form"):
    text = st.text_area("Enter text:", "What is a Retrieval-Augmented system?")
    submitted = st.form_submit_button("Submit")

    if submitted:    
        answer, citations = process_inputs({"question": text})
        citation_text= '\n- '.join([title for title in citations])

        response = f"""
{answer}

Citations:
- {citation_text}
"""
        st.info(response)

