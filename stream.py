import base64
import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama  
import os
import time
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_elasticsearch.vectorstores import ElasticsearchStore, DenseVectorStrategy

st.set_page_config(page_title="Akili", page_icon="🅰️", layout="wide")

# Header and logo functions
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def load_css():
    with open("static/styles.css") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

# Load CSS
load_css()

img_path = "static/main-logo.png"
img_base64 = img_to_base64(img_path)

# Create header container with CSS
header = st.container()
header.write(f"""
    <div class='fixed-header'>
        <img src="data:image/png;base64,{img_base64}" class="logo">
        <h1>Akili BotZet</h1>
    </div>
""", unsafe_allow_html=True)

GOOGLE_API_KEY = "AIzaSyANXuJrBgTaReoX5yU040oSOSMzFAZNEGI"

# Hide Streamlit style
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Custom CSS
st.markdown(
    """
    <style>
        .st-emotion-cache-vj1c9o {
            background-color: rgb(242, 242, 242, 0.68);
        }
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 3;
            background-color: rgb(242, 242, 242, 0.68);
            z-index: 999;
            text-align: center;
        }
        .fixed-header {
            border-bottom: 0;
        }
        .st-emotion-cache-7tauuy {
            width: 100%;
            padding: 6.5rem 3.5rem 3.5rem;
            min-width: auto;
            max-width: initial;
            margin-top: -50px;
        }
        
        .st-emotion-cache-1whx7iy p {
            word-break: break-word;
            margin-bottom: 0px;
            font-size: 12px;
            font-weight: 500;
        }
        .st-bi {
            border-bottom-color: black;
        }
        .st-bh {
            border-top-color:black;
        }
        .st-bg {
            border-right-color: black;
        }
        .st-bf {
            border-left-color: black;
        }
        .st-emotion-cache-1ghhuty {
            display: flex;
            width: 2rem;
            height: 2rem;
            flex-shrink: 0;
            border-radius: 0.5rem;
            -webkit-box-align: center;
            align-items: center;
            -webkit-box-pack: center;
            justify-content: center;
            background-color: navy;
            color: rgb(255, 255, 255);
        }
        
        .st-emotion-cache-bho8sy {
            display: flex;
            width: 2rem;
            height: 2rem;
            flex-shrink: 0;
            border-radius: 0.5rem;
            -webkit-box-align: center;
            align-items: center;
            -webkit-box-pack: center;
            justify-content: center;
            background-color: black;
            color: rgb(255, 255, 255);
        }
        
        .st-emotion-cache-uzeiqp p {
            word-break: break-word;
            text-align: justify;
        }
        .st-b0 {
            border-bottom-color: black;
        }
        .st-az {
            border-top-color: black;
        }
        .st-ay {
            border-right-color: black;
        }
        .st-ax {
            border-left-color: black;
        }
        
        .st-emotion-cache-1jicfl2 {
            width: 100%;
            padding: 6rem 1rem 10rem;
            min-width: auto;
            max-width: initial;
            margin-top: -100px;
        }
            
        [data-testid="stSidebarContent"]{
            background: #007bff;
        }
        
        .st-b1 {
            background-color:rgb(242, 242, 242, 0.68);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar logo
image_path = "static/alkaili_logo.png"
with open(image_path, "rb") as f:
    image_bytes = f.read()
image_base64 = base64.b64encode(image_bytes).decode()

st.sidebar.markdown(f"""
<div style="position : center; margin-left: 50px; margin-bottom: 20px;  z-index: 9999;">
    <div style="color:black; padding-top: -30px; padding-bottom: -10px; border: 0px solid #ccc; border-radius: 5px; max-height:300px; overflow: hidden; width:150px;">
        <img src="data:image/jpeg;base64,{image_base64}" alt="Your Image" style="width: 100%; height: auto;">
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def generate_response(es_cloud_id, es_api_key):
    embedding = HuggingFaceEmbeddings()

    test = DenseVectorStrategy(
        hybrid=True,
        rrf=False,
        text_field="text"
    )
    
    vector_db = ElasticsearchStore(
        embedding=embedding,
        index_name="akili_total",
        es_cloud_id=es_cloud_id,
        es_api_key=es_api_key,
        strategy=test,
    )
    
    prompt_template = """You are an advanced AI designed to analyze and interpret unstructured documents in multiple formats. With extensive expertise in government regulations, policies, and official language, you meticulously assess user inquiries to ensure complete comprehension. Correct any spelling mistakes internally before crafting your response. Your answers must strictly adhere to the content and context of the documents, remaining concise and focused. Do not reference or mention the names of source files. If no pertinent information is available, simply respond with "I don't know."

    {context}

    Question: {question}
    Helpful Answer:"""

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAPI_KEY"])
    retriever = vector_db.as_retriever()

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
    )

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa

def main():
    spelling_template = """
    You are a helpful assistant tasked with correcting any spelling mistakes in the given query. If there are no spelling mistakes, return the original query unchanged. If there are spelling mistakes, provide the corrected query.

    Original query: {question}

    Corrected query:
    """
    spelling_llm = ChatOpenAI(model="gpt-4o", api_key=st.secrets["OPENAPI_KEY"])
    spelling_prompt = ChatPromptTemplate.from_template(spelling_template)

    spelling_chain = spelling_prompt | spelling_llm | StrOutputParser()

    chain = generate_response(
        es_cloud_id=st.secrets["es_cloud_id"],
        es_api_key=st.secrets["es_api_key"]
    )
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if 'sources' in message:
                st.markdown("<div class='source-links'><b>Sources:</b> " + 
                            ", ".join([f"<a href='https://akilisa.sharepoint.com/sites/SAGazetteHub/shared Documents/Genie Documents_5498/{source}' target='_blank'>{source}</a>" 
                                       for source in message['sources']]) + 
                            "</div>", unsafe_allow_html=True)
                
    
    query = st.chat_input("Enter Your Query here:")
    
    with st.sidebar:
        st.subheader("Here are some sample prompts:👇🏼")

        sample_prompts = [
            "How do I provide feedback on the proposed import/export law amendments?",
            "How do I comply with new vehicle registration regulations?",
            "What is the publication date of the government gazette in Parliament concerning the Department of Labour?",
            "Who are the members of the Executive Council responsible for Cooperative Governance and Traditional Affairs?",
            "What changes have been made to compliance in health and safety regulations?",
            "Can you summarise government announcements related to healthcare?",
            "What are the compliance requirements for environmental impact assessments?",
            "How does this law immigration amendment bill 2018 affect my rights as a citizen?",
            "Are there any consultations related to climate change policy?",
            "How does the new procurement regulation impact departmental purchasing?",
            "Where can I find the full route descriptions of the submitted permit applications?",
            "What information is included in an application for a cross-border transport permit?",
            "What actions have been taken by the Minister of Finance regarding the delegation of powers and forfeiture of funds, and what are the specifics of the forfeiture decision?",
            "What is the proposed amendment to the Civil Aviation Technical Standards regarding Remotely Piloted Aircraft Systems, and how can interested parties submit comments?",
            "What are the details of the recent transport permit applications for cross-border services between South Africa and Zimbabwe?",
            "What are the key components and goals of the White Paper on Conservation and Sustainable Use of South Africa's Biodiversity approved by Cabinet on 29 March 2023?",
            "When will the new scale of fees for medical aid under section 76 of the Compensation for Occupational Injuries and Diseases Act take effect?",
            "Who is the Minister of Employment and Labour who gave notice regarding the scale of fees under the Compensation for Occupational Injuries and Diseases Act?"
        ]

        for prompt in sample_prompts:
            if st.button(prompt):
                query = prompt

    if query:
        print(query)
        query = spelling_chain.invoke({"question": query})
        print(query)
        if query:
            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("Thinking..."):
                st.session_state.messages.append({"role": "user", "content": query})
                result = chain({"query": query})
                print(result)
                response = result['result']
                
                if any(phrase in response.lower() for phrase in ["don't know", "do not know", "i don't know", "i do not know"]):
                    response = "No Relevant Documents Found"
                
                sources = []
                if response != "No Relevant Documents Found":
                    top_n = 3
                    unique_sources = set()
                    document_count = 0
                    
                    for doc in result['source_documents']:
                        source = doc.metadata['source']
                        if source not in unique_sources:
                            unique_sources.add(source)
                            document_count += 1
                            filename = os.path.basename(source.replace('\\', '/').split('/')[-1])
                            # filename = os.path.basename(source)
                            st.write(filename)
                            if "_extracted" in filename:
                                index = filename.find("_extracted")
                                filename = filename[:index] + filename[index + len("_extracted"):]
                            filename = filename.replace('.txt', '.pdf')
                            sources.append(filename)
                            
                            if document_count == top_n:
                                break
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                    if sources:
                        st.markdown("<div class='source-links'><b>Sources:</b><br>" + 
                                    "<br>".join([f"<a href='https://akilisa.sharepoint.com/sites/SAGazetteHub/shared Documents/Genie Documents_5498/{source}' target='_blank'>{source}</a>" 
                                                for source in sources]) + 
                                    "</div>", unsafe_allow_html=True)

                st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})

if __name__ == '__main__':
    main()