# ==============================================================================
# FELIX AI SOLUTIONS - DEMO "DOBLE IMPACTO"
# VERSIÓN 1.0 - Creado por: Tu Nombre, Mentor: Gemini
# ==============================================================================

# --- 1. IMPORTACIÓN DE LIBRERÍAS ---
import streamlit as st
from openai import OpenAI
import chromadb
import pandas as pd # Usaremos pandas para el gráfico de barras

# --- 2. CONFIGURACIÓN INICIAL DE LA APLICACIÓN ---

# Configuración de la página (título, ícono, layout)
st.set_page_config(
    page_title="Demo Candidato Digital - Felix AI",
    page_icon="🤖",
    layout="wide"
)

# Título principal de la aplicación
st.title("🤖 Demo Candidato Digital: La Plataforma de Inteligencia de Campaña")
st.markdown("---")


# --- 3. FUNCIÓN DE CARGA DEL "CEREBRO" (CON CACHÉ) ---
# Usamos el caché de Streamlit para que esta función pesada solo se ejecute una vez.
@st.cache_resource
def cargar_cerebro_candidato():
    print("INICIANDO CARGA DEL CEREBRO (esto solo debería aparecer una vez)...")
    
    # Aquí pegamos el documento COMPLETO. Es la única fuente de verdad.
    campaign_document = """
    PEGA AQUÍ TU DOCUMENTO COMPLETO Y ACTUALIZADO CON LOS DOBLES SALTOS DE LÍNEA
    """
    text_chunks = campaign_document.split('\n\n')
    text_chunks = [chunk.strip() for chunk in text_chunks if chunk.strip()]

    # Inicializamos ChromaDB localmente
    client_chroma = chromadb.EphemeralClient()
    collection = client_chroma.get_or_create_collection("candidato_ia") # Usamos get_or_create para evitar errores si se recarga
    
    # Creamos y cargamos los embeddings si la colección está vacía
    if collection.count() == 0:
        embed_model = "text-embedding-ada-002"
        # Usamos el cliente de OpenAI que se configurará más adelante
        res = st.session_state.openai_client.embeddings.create(input=text_chunks, model=embed_model)
        embeddings_list = [item.embedding for item in res.data]
        doc_ids = [f"doc_chunk_{i}" for i in range(len(text_chunks))]
        collection.add(embeddings=embeddings_list, documents=text_chunks, ids=doc_ids)

    print(f"CARGA DEL CEREBRO COMPLETADA. Total de vectores: {collection.count()}")
    return collection

# --- 4. LÓGICA DE CONVERSACIÓN ---
def ask_candidato_ia(pregunta, collection):
    contexto = "" # Inicializamos el contexto
    # 1. Crear embedding para la pregunta
    try:
        res_query = st.session_state.openai_client.embeddings.create(
            input=pregunta, model="text-embedding-ada-002"
        )
        query_embedding = res_query.data[0].embedding
        
        # 2. Buscar en ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding], n_results=3
        )
        contexto = "\n\n".join(results['documents'][0])
    except Exception as e:
        st.error(f"Error al buscar en la base de datos: {e}")
        return "Error al buscar información."

    # 3. Construir el prompt
    prompt_template = f"""
    Eres el asistente digital de Javier Montoya. Tu misión es responder preguntas basándote ESTRICTA Y ÚNICAMENTE en la siguiente información oficial:
    ---
    {contexto}
    ---
    PREGUNTA: "{pregunta}"
    Si la información no es suficiente para responder, di EXACTAMENTE: "Esa es una excelente pregunta. No tengo la información detallada sobre ese punto, lo consultaré con el equipo de campaña."
    RESPUESTA:
    """
    
    # 4. Generar la respuesta final
    try:
        res_completion = st.session_state.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asistente de campaña política servicial y preciso."},
                {"role": "user", "content": prompt_template}
            ]
        )
        return res_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error al generar la respuesta: {e}")
        return "Error al generar la respuesta."

# --- 5. INICIALIZACIÓN DEL ESTADO DE LA SESIÓN ---
# El estado de la sesión guarda variables mientras el usuario interactúa con la app
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'temas' not in st.session_state:
    st.session_state.temas = {"Seguridad": 0, "Empleo": 0, "Turismo": 0, "Agro": 0, "Corrupción": 0}
if 'oportunidades' not in st.session_state:
    st.session_state.oportunidades = []

# --- 6. BARRA LATERAL PARA LA CONFIGURACIÓN ---
with st.sidebar:
    st.header("Configuración")
    api_key_input = st.text_input("Introduce tu Clave API de OpenAI", type="password")
    if st.button("Conectar"):
        if api_key_input:
            st.session_state.openai_client = OpenAI(api_key=api_key_input)
            st.success("¡Conectado a OpenAI!")
            # Cargamos el cerebro una vez que tenemos el cliente de OpenAI
            with st.spinner("Cargando cerebro del candidato..."):
                st.session_state.collection = cargar_cerebro_candidato()
            st.success("¡Cerebro cargado y listo para conversar!")
        else:
            st.warning("Por favor, introduce una clave API.")

# --- 7. LÓGICA DE LA INTERFAZ PRINCIPAL ---
if not st.session_state.openai_client:
    st.info("👋 ¡Bienvenido! Por favor, introduce tu clave API de OpenAI en la barra lateral para comenzar.")
else:
    # Definimos las dos columnas para el layout "Doble Impacto"
    col1, col2 = st.columns([2, 1]) # Columna del chat 2/3, columna del panel 1/3

    # === Columna 1: La Interfaz de Chat ===
    with col1:
        st.subheader("📱 Chat con el Candidato Digital")

        # Muestra los mensajes del historial
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input del usuario en la parte inferior
        if prompt := st.chat_input("Escribe tu pregunta aquí..."):
            # Añade el mensaje del usuario al historial y a la pantalla
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Muestra un spinner mientras el asistente piensa
            with st.spinner("Pensando..."):
                # Obtiene la respuesta del asistente
                response = ask_candidato_ia(prompt, st.session_state.collection)

                # Añade la respuesta del asistente al historial y a la pantalla
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

                # Actualiza el panel de control
                # Lógica de temas (simple conteo de palabras clave)
                if any(keyword in prompt.lower() for keyword in ["seguridad", "policía", "drones", "hurto"]):
                    st.session_state.temas["Seguridad"] += 1
                if any(keyword in prompt.lower() for keyword in ["empleo", "trabajo", "jóvenes", "emprendimiento"]):
                    st.session_state.temas["Empleo"] += 1
                if any(keyword in prompt.lower() for keyword in ["turismo", "salento", "filandia", "viajes"]):
                    st.session_state.temas["Turismo"] += 1
                if any(keyword in prompt.lower() for keyword in ["agro", "campo", "cafeteros", "agricultores"]):
                    st.session_state.temas["Agro"] += 1
                if any(keyword in prompt.lower() for keyword in ["corrupción", "transparencia", "robar"]):
                    st.session_state.temas["Corrupción"] += 1
                
                # Lógica de oportunidades
                if "No tengo la información detallada" in response:
                    st.session_state.oportunidades.append(prompt)

                # Forzamos la recarga de la app para que el panel se actualice
                st.rerun()

    # === Columna 2: El Panel de Control del Candidato ===
    with col2:
        st.subheader("📊 Panel de Inteligencia")
        st.markdown("---")
        
        # Módulo 1: Temas de Interés
        st.write("**Temas de Interés del Votante**")
        if sum(st.session_state.temas.values()) > 0:
            df_temas = pd.DataFrame(list(st.session_state.temas.items()), columns=['Tema', 'Menciones'])
            st.bar_chart(df_temas.set_index('Tema'))
        else:
            st.info("El gráfico aparecerá cuando comience la conversación.")
        st.markdown("---")

        # Módulo 2: Oportunidades
        st.write("**Preguntas sin Respuesta / Oportunidades**")
        if st.session_state.oportunidades:
            st.text_area("El votante preguntó sobre temas no cubiertos:", 
                         value="- " + "\n- ".join(st.session_state.oportunidades), 
                         height=200, 
                         disabled=True)
        else:
            st.info("Aquí aparecerán las preguntas que el asistente no pudo responder.")