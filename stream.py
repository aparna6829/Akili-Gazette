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
            background-color:none;
        }

        /* Additional CSS for tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #ffffff;
            border-radius: 4px;
            color: #000000;
            padding: 8px 16px;
        }

        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #0066cc;
            color: #ffffff;
        }

        .sidebar-prompt {
            margin-bottom: 8px;
            padding: 8px;
            border-radius: 4px;
            background-color: #f0f2f6;
            cursor: pointer;
        }

        .sidebar-prompt:hover {
            background-color: #e0e2e6;
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
    
    prompt_template ="""You are an AI assistant specialized in analyzing and extracting precise information from official documents and regulations. Follow these strict guidelines to ensure accuracy and clarity:

        1. Context Analysis

        Carefully examine the context provided in the question.
        Focus on locating specific, relevant information from all referenced documents.
        Consider both direct and indirect relationships within the documents to enrich the answer.
        2. Response Formatting

        Present answers clearly and concisely.
        Use bullet points for multiple details or instructions.
        Include exact details where applicable for clarity.
        3. Accuracy Standards

        Rely solely on information present within the provided context.
        Avoid assumptions or external knowledge.
        If only partial information is available, acknowledge this limitation.
        4. Special Instructions

        Do not reference specific document titles or names.
        Maintain relevance and a professional tone.
        Present the response strictly in bullet points for ease of reading.
        Respond only to specific, relevant questions. If the answer is unknown, reply with “I don’t know” without speculation.







    Context: {context}

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
        st.subheader("Sample Prompts by Category:👇🏼")
        
        # Create tabs for different categories
        tab1, tab2, tab3, tab4, tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs(["Legal Practitioners", "Government Officials", "Business Compliance Officers", "Journalists and Media Professionals", "Researchers and Academics","NGO Representatives","Procurement Specialists and Contractors","Tax Professionals and Accountants","Environmental Consultants","Trade Union Representatives"])
        
        with tab1:
            st.markdown("### Legal Practitioners sample prompts")
            regulatory_prompts = [
                "What are the latest amendments to the Promotion of Access to Information Act?",
                "Recent case law decisions impacting contract enforcement in South Africa.",
                "Updates on regulations governing electronic signatures.",
                "How is the SKAO Intellectual Property Policy approved and amended?",
                "Newly published legal notices affecting maritime law.",

            ]
            for prompt in regulatory_prompts:
                if st.button(prompt, key=f"reg_{prompt}"):
                    query = prompt
        
        with tab2:
            st.markdown("### Government Officials sample prompts")
            permit_prompts = [
                "Recent policies regarding adjustments to public service salaries.",
                "Latest guidelines on municipal financial management.",
                "Updates to the National Development Plan initiatives.",
                "Recent proclamations regarding public holidays.",
                "Changes in intergovernmental relations legislation."

            ]
            for prompt in permit_prompts:
                if st.button(prompt, key=f"permit_{prompt}"):
                    query = prompt
        
        with tab3:
            st.markdown("### Business Compliance Officers sample prompts")
            policy_prompts = [
                "New compliance requirements under the Consumer Protection Act.",
                "Updates on anti-money laundering regulations.",
                "Changes to the Occupational Health and Safety Act affecting manufacturing.",
                "Latest B-BBEE compliance codes for medium enterprises.",
                "Environmental compliance deadlines for emissions reporting.",

            ]
            for prompt in policy_prompts:
                if st.button(prompt, key=f"policy_{prompt}"):
                    query = prompt
        
        with tab4:
            st.markdown("### Journalists and Media Professionals sample prompts")
            feedback_prompts = [
                "Announcements on upcoming parliamentary sessions.",
                "Recent government responses to public protests.",
                "Updates on legislation affecting freedom of the press.",
                "Notices about changes in broadcasting regulations.",
                "Information on newly appointed cabinet ministers."

            ]
            for prompt in feedback_prompts:
                if st.button(prompt, key=f"feedback_{prompt}"):
                    query = prompt
        
        with tab5:
            st.markdown("### Researchers and Academics sample prompts")
            announcement_prompts = [
                "Statistical releases on national unemployment rates.",
                "Historical data on education policy reforms since 2000.",
                "Recent government reports on climate change impact.",
                "Updates on research funding opportunities from the Department of Science and Innovation.",
                "Analysis of urbanization trends published in the gazette."

            ]
            for prompt in announcement_prompts:
                if st.button(prompt, key=f"announce_{prompt}"):
                    query = prompt
        
        with tab6:
            st.markdown("### NGO Representatives sample prompts")
            announcement_prompts = [
                "Latest policies affecting non-profit registration.",
                "Updates on social grant distributions.",
                "Government notices on human trafficking legislation.",
                "Announcements of public participation meetings on environmental issues.",
                "Changes to laws impacting refugee rights."

            ]
            for prompt in announcement_prompts:
                if st.button(prompt, key=f"announce_{prompt}"):
                    query = prompt
                    
        with tab7:
            st.markdown("### Procurement Specialists and Contractors sample prompts")
            announcement_prompts = [
                "New tender opportunities in the healthcare sector.",
                "Updates to procurement policies under the Public Finance Management Act.",
                "Notices of awarded contracts for infrastructure projects.",
                "Changes in supplier accreditation requirements.",
                "Upcoming bids for renewable energy projects.",

            ]
            for prompt in announcement_prompts:
                if st.button(prompt, key=f"announce_{prompt}"):
                    query = prompt
        
        with tab8:
            st.markdown("### Tax Professionals and Accountants sample prompts")
            announcement_prompts = [
                "Recent amendments to the Income Tax Act.",
                "Updates on tax compliance regulations for trusts.",
                "Changes in customs and excise duties announced.",
                "Notices about tax relief measures for businesses.",
                "New SARS guidelines on cryptocurrency taxation."

            ]
            for prompt in announcement_prompts:
                if st.button(prompt, key=f"announce_{prompt}"):
                    query = prompt
        with tab9:
            st.markdown("### Environmental Consultants sample prompts")
            announcement_prompts = [
                "Updates on environmental impact assessment regulations.",
                "New protected areas declared by the government.",
                "Changes to the National Water Act affecting resource management.",
                "Latest waste management policies and compliance deadlines.",
                "Announcements on air quality standards revisions.",

            ]
            for prompt in announcement_prompts:
                if st.button(prompt, key=f"announce_{prompt}"):
                    query = prompt
                    
                    
        with tab10:
            st.markdown("### Trade Union Representatives sample prompts")
            announcement_prompts = [
               "Adjustments to the national minimum wage.",
                "Updates on labor law amendments affecting collective bargaining.",
                "Notices regarding changes to worker safety regulations.",
                "Recent developments in employment equity legislation.",
                "Government responses to proposed industrial action."

            ]
            for prompt in announcement_prompts:
                if st.button(prompt, key=f"announce_{prompt}"):
                    query = prompt

    if query:
        # print(query)
        query = spelling_chain.invoke({"question": query})
        # print(query)
        if query:
            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("Processing Your Query..."):
                st.session_state.messages.append({"role": "user", "content": query})
                result = chain({"query": query})
                # print(result)
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