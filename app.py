import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loaders import *

TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Video', 'Pdf', 'Csv', 'Txt'
]

CONFIG_MODELOS = {'Groq': 
                        {'modelos': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
                         'chat': ChatGroq},
                  'OpenAI': 
                        {'modelos': ['gpt-4o-mini', 'gpt-4o', 'o1-preview', 'o1-mini','gpt-5-nano'],
                         'chat': ChatOpenAI}}

MEMORIA = ConversationBufferMemory()

def carrega_todos_arquivos(arquivos_pdf, arquivos_csv, arquivos_txt, sites, videos_youtube):
    documentos = []
    # PDFs
    for pdf in arquivos_pdf or []:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(pdf.read())
            documentos.append(carrega_pdf(temp.name))
    # CSVs
    for csv in arquivos_csv or []:
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(csv.read())
            documentos.append(carrega_csv(temp.name))
    # TXTs
    for txt in arquivos_txt or []:
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(txt.read())
            documentos.append(carrega_txt(temp.name))
    # Sites
    for site in sites or []:
        if site.strip():
            documentos.append(carrega_site(site.strip()))
    # YouTube
    for video in videos_youtube or []:
        if video.strip():
            documentos.append(carrega_youtube(video.strip()))
    return '\n\n'.join(documentos)

def carrega_modelo_multi(provedor, modelo, api_key, arquivos_pdf, arquivos_csv, arquivos_txt, sites, videos_youtube):
    documento = carrega_todos_arquivos(arquivos_pdf, arquivos_csv, arquivos_txt, sites, videos_youtube)

    system_message = f'''Você é um assistente amigável chamado Oráculo de testes.
    Você possui acesso às seguintes informações vindas de múltiplos documentos:

    ####
    {documento}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substita por S.

    Se a informação do documento for algo como 
    "Just a moment...Enable JavaScript and cookies to continue", 
    sugira ao usuário carregar novamente o Oráculo!'''

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key, temperature=1)
    chain = template | chat

    st.session_state['chain'] = chain

def pagina_chat():
    st.markdown(
        """
        <h2 style="display:flex; align-items:center; gap:10px;">
            🤖 <span>Bem-vindo ao <b>Oráculo de testes</b></span>🪲
        </h2>
        <hr style="margin-top:5px; margin-bottom:15px; border:1px solid #5f3c3c;">
        """,
        unsafe_allow_html=True
    )

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Carregue o Oráculo de teste')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Fale com o oráculo de testes')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario, 
            'chat_history': memoria.buffer_as_messages
        }))
        
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    with tabs[0]:
        arquivos_pdf = st.file_uploader('Faça upload de PDFs', type=['pdf'], accept_multiple_files=True)
        arquivos_csv = st.file_uploader('Faça upload de CSVs', type=['csv'], accept_multiple_files=True)
        arquivos_txt = st.file_uploader('Faça upload de TXTs', type=['txt'], accept_multiple_files=True)
        sites = st.text_area('URLs de sites (um por linha)').splitlines()
        videos_youtube = st.text_area('URLs de vídeos (um por linha)').splitlines()
    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor dos modelo', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a api key para o provedor {provedor}',
            value=st.session_state.get(f'api_key_{provedor}'))
        st.session_state[f'api_key_{provedor}'] = api_key
    
    if st.button('Inicializar Oráculo', use_container_width=True):
        carrega_modelo_multi(
            provedor, modelo, api_key,
            arquivos_pdf, arquivos_csv, arquivos_txt, sites, videos_youtube
        )
    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

    st.markdown(
        """
        <div style="
            margin-top: 20px;
            padding: 10px;
            font-size: 12px;
            color: white;
            text-align: justify;
            border: 1px dashed #888;
            border-radius: 6px;
        ">
        Segundo o <b>INTERNATIONAL SOFTWARE TESTING QUALIFICATIONS BOARD (2024)</b>, 
        um oráculo de teste é uma fonte de informação que determina os resultados esperados...
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == '__main__':
    main()