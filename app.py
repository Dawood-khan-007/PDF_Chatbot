import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
nltk.download('punkt')



with st.sidebar:
    st.title('Welcome to LLM Chat App🤗💬')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    This app can be used for extracting information from the text document:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Dawood Ahmad Khan')

def summarize_text(text, num_sentences=6):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return ' '.join([str(sentence) for sentence in summary])


def main():
    st.header("PDF Question Answer Session👩‍💻")
    load_dotenv()
    pdf = st.file_uploader("Please Upload Your PDF File📩 and click 'Process'", type='pdf')

    if pdf is not None:
        pdf_file_reader = PdfReader(pdf)

        text = ""
        for page in pdf_file_reader.pages:
            text += page.extract_text()
        
        summarized_text = summarize_text(text)
        st.subheader("Summary of The Document:")
        st.write(summarized_text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
        
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        st.header("Ask questions about your PDF file🤔")
        query = st.text_input("Enter Your Question: ")
        st.write(query)

    if st.button("Process"):
        with st.spinner("Processing"):
            if query:
                docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(openai_api_key=st.secrets["openai_api_key"],streaming=True)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            
            
            if response:
                st.subheader("Answer:")
                st.write(response)
            else:
                st.warning("No answer found for the given question.")

    
    
                    
if __name__ == '__main__':
    main()

