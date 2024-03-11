import streamlit as st
import os
import feedparser
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import LanceDB
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#Configuramos el Streamlit
st.set_page_config('PDFChat', page_icon=":page_facing_up")
st.header("Bienvenidos al Mundo de la IA 🤖", divider='rainbow')

st.subheader('_Analizador de PDF_ :sunglasses:')


os.environ['OPENAI_API_KEY'] = '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'

model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

obj_pdf = st.file_uploader("Carga tu Documento", type="pdf", on_change=st.cache_resource.clear)
@st.cache_resource
def crear_embeddings(pdf):
    leer_pdf = PdfReader(pdf)
    text = ""
    for pag in leer_pdf.pages:
        text += pag.extract_text()

    #Divimos en chunks
    dividir_text = RecursiveCharacterTextSplitter(
        chunk_size= 800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = dividir_text.split_text(text)

    #se crea los embeddings y se almacenan en una base
    crear_embeddings = OpenAIEmbeddings()
    bd_embedding = LanceDB.from_texts(chunks, crear_embeddings)

    return bd_embedding, text

if obj_pdf:
    bd_embeddings, texto_documento = crear_embeddings(obj_pdf)
    u_pregunta = st.text_input("¿Cuales es la pregunta?")

    if st.button("Responde"):
        #Realiza la busqueda de similitud y procesa la pregunta
        docs = bd_embeddings.similarity_search(u_pregunta, 3)
        llm = OpenAI(model_name='gpt-3.5-turbo-instruct')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=u_pregunta)
        st.write("Respuesta Generada: ", respuesta)

        with get_openai_callback() as cost:
            response = chain.invoke(input={"question": u_pregunta, "input_documents": docs})
            print(cost)

            #st.write(response["output_text"])
            st.write("Costo de la Operacion:", cost)

st.subheader('_Analizador de Oraciones_ 🖊️', divider='rainbow')


oracion = st.text_area("Ingresa las Oraciones")
etiquetas_personalizadas = st.text_input("ingresa etiquetas separadas por comas:")

if st.button("clasificador de oraciones :sunglasses:"):
    oraciones_lista = oracion.split("\n")

    for oracion in oraciones_lista:
        premise = oracion
        hypothesis = etiquetas_personalizadas

        input = tokenizer(premise, hypothesis, return_tensors="pt")
        output = model(**input)

        prediction = output["logits"][0].softmax(dim=0)
        prediction = {name: round(float(pred) * 100, 1) for pred, name in
                      zip(prediction, etiquetas_personalizadas.split(","))}
        st.write(f"Resultado para '{oracion}':", prediction)




