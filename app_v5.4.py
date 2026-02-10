"""
Medical Affairs AI Agent v5.4
=============================

Ein LangGraph-basierter Medical-Information-Agent zur automatisierten 
Klassifikation und Beantwortung von Anfragen im Pharmakontext.

Features:
---------
- Automatische Triage von Anfragen (Nebenwirkungsmeldung, medizinische Info, Sonstiges)
- RAG-basierte Antwortgenerierung mit Dokumenten-Grading
- Iterative Qualitätskontrolle durch Critique-Loop
- Fallback-Modus bei fehlenden Quelldokumenten
- Vollständiges Audit-Logging mit Timestamps

Abhängigkeiten:
---------------
- LangGraph für State-Machine-Workflow
- LangChain für LLM-Integration und Retrieval
- Streamlit für Web-Interface
- Chroma für Vektordatenbank

Autor: [Dein Name]
Datum: 2025-12-19
"""

import os
import datetime
from typing import TypedDict, List

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# === DIAGNOSE BLOCK START ===
import google.generativeai as genai

try:
    # Key laden
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    
    st.sidebar.error("🔍 API-DIAGNOSE LÄUFT...")
    
    # Modelle abfragen, die Embeddings können
    models = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
    
    if not models:
        st.sidebar.error("❌ KEINE Embedding-Modelle gefunden! API deaktiviert?")
    else:
        st.sidebar.success(f"✅ Verfügbare Modelle: {models}")
except Exception as e:
    st.sidebar.error(f"❌ Diagnose fehlgeschlagen: {e}")
# === DIAGNOSE BLOCK ENDE ===


# ==============================================================================
# MULTI-LANGUAGE TEMPLATES
# ==============================================================================

TEMPLATES = {
    "DE": {
        "header": "vielen Dank für Ihre Nachricht.",
        "ae_intro": "Wir haben Ihre Meldung bezüglich einer Nebenwirkung erhalten und nehmen Ihr Anliegen äußerst ernst. Ich habe den Sachverhalt persönlich an unsere Abteilung für Arzneimittelsicherheit (Pharmacovigilance) weitergeleitet.",
        "ae_transition": "Zu Ihrer inhaltlichen Frage kann ich Ihnen Folgendes mitteilen:",
        "fallback": "⚠️ HINWEIS: Die folgenden Informationen basieren auf allgemeinem medizinischen Wissen und sind nicht durch unsere interne Datenbank gedeckt.",
        "footer": "Mit freundlichen Grüßen,\nDr. Eike Bent Preuß | Medical Affairs Manager",
        "salutation_fallback": "Sehr geehrte Damen und Herren,"
    },
    "EN": {
        "header": "thank you for your message.",
        "ae_intro": "We have received your report regarding a potential adverse event and take it very seriously. I have forwarded this matter to our Pharmacovigilance department for documentation and review.",
        "ae_transition": "Regarding your medical inquiry, I can provide the following information:",
        "fallback": "⚠️ NOTE: The following information is based on general medical knowledge and is not covered by our internal database.",
        "footer": "Sincerely,\nDr. Eike Bent Preuß | Medical Affairs Manager",
        "salutation_fallback": "Dear Sir or Madam,"
    }
}

# ==============================================================================
# CUSTOM CSS STYLING
# ==============================================================================

def apply_custom_css():
    """
    Wendet benutzerdefiniertes CSS für ein professionelles medizinisches Design an.
    
    Features:
    - Medizinisches Farbschema (Blau/Grün-Töne)
    - Verbesserte Buttons mit Hover-Effekten
    - Optimierte Cards und Expander
    - Professionelle Typografie
    """
    st.markdown("""
        <style>
        /* Hauptcontainer */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8f0f7 100%);
        }
        
        /* Header Styling */
        h1 {
            color: #1e3a8a;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #3b82f6;
        }
        
        h2 {
            color: #1e40af;
            font-family: 'Helvetica Neue', sans-serif;
            margin-top: 2rem;
        }
        
        h3 {
            color: #2563eb;
            font-family: 'Helvetica Neue', sans-serif;
        }
        
        /* Text Area Styling */
        .stTextArea textarea {
            border: 2px solid #cbd5e1;
            border-radius: 8px;
            padding: 1rem;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        .stTextArea textarea:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
        }
        
        /* Download Button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
        }
        
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(16, 185, 129, 0.3);
        }
        
        /* Info/Warning/Success Boxes */
        .stAlert {
            border-radius: 8px;
            border-left: 4px solid;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
            border-radius: 8px;
            padding: 0.75rem;
            font-weight: 600;
            color: #1e3a8a;
            border: 1px solid #cbd5e1;
        }
        
        .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p {
            color: white;
        }
        
        /* Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #1e3a8a;
            font-weight: 700;
        }
        
        /* Caption Styling für Quellen */
        .stCaption {
            background: #f8fafc;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            border-left: 3px solid #3b82f6;
            margin: 0.25rem 0;
            font-family: 'Courier New', monospace;
        }
        
        /* Spinner Styling */
        .stSpinner > div {
            border-top-color: #3b82f6 !important;
        }
        
        /* Toast Notification */
        .stToast {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border-radius: 8px;
        }
        
        /* Code Block Styling */
        .stMarkdown pre {
            background: #1e293b;
            color: #f8fafc; /* NEU: Helle Textfarbe für Kontrast */
            border-radius: 8px;
            padding: 1rem;
            border-left: 4px solid #3b82f6;
        }
        
        /* NEU: Sicherstellen, dass auch das innere Code-Element hell ist */
        .stMarkdown pre code {
            color: #f8fafc;
        }
        
        /* Log Text Styling */
        .stText {
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            color: #334155;
        }
        </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# 1. KONFIGURATION & SETUP
# ==============================================================================

PAGE_TITLE = "Demo: MedAffairs AI Agent v5.4"
DB_FOLDER = "chroma_db"  # Pfad zur Chroma-Vektordatenbank
REPORT_FOLDER = "reports"  # Ordner für Log-Dateien

# Standardfrage für Testzwecke
DEFAULT_QUESTION = (
    "Sehr geehrter Herr Dr. Preuß,\n\n"
    "Wir haben einem Säugling Espumisan gegeben. Kurz darauf bekam das Kind Atemnot. Sind diese Nebenwirkungen bekannt?\n\n"
    "Bitte um Rückmeldung.\n\n"
    "Mit freundlichen Grüßen,\n"
    "Dr. Anna Müller"
)

# Streamlit-Konfiguration
st.set_page_config(
    page_title=PAGE_TITLE, 
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# Custom CSS anwenden
apply_custom_css()

# Umgebungsvariablen laden (z.B. OPENAI_API_KEY)
load_dotenv()

# Verzeichnisse sicherstellen
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Prüfung, ob Datenbank existiert
if not os.path.exists(DB_FOLDER):
    st.error("❌ Datenbank nicht gefunden! Bitte erst 'indexer.py' ausführen.")
    st.stop()

# LLM-Instanz mit Gemini-Modell initialisieren
llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it", 
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
)


# Vektordatenbank und Retriever initialisieren
#embeddings = OpenAIEmbeddings()
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
)
vectorstore = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Top-3 Dokumente


# ==============================================================================
# 2. STATE DEFINITION
# ==============================================================================

class AgentState(TypedDict):
    """
    Zentrale State-Struktur für den LangGraph-Workflow.
    
    Attributes:
        question: Die ursprüngliche Nutzer-Anfrage
        category: Klassifizierte Kategorie (ADVERSE_EVENT_ONLY, HYBRID, MEDICAL_INFO, OTHER)
        context: Zusammengeführter Text aus relevanten Dokumenten
        documents: Liste der geladenen Document-Objekte
        source_names: Dateinamen der verwendeten Quellen
        draft: Generierter E-Mail-Entwurf
        critique: Feedback vom Critique-Node (PASS oder Fehlerbeschreibung)
        revision_count: Anzahl der Draft-Iterationen
        logs: Liste aller Workflow-Logs mit Timestamps
        fallback_mode: True, wenn keine DB-Dokumente gefunden wurden
        has_ae_component: True, wenn Adverse-Event-Komponente erkannt wurde
    """
    question: str
    category: str
    context: str
    documents: List[Document]
    source_names: List[str]
    draft: str
    critique: str
    revision_count: int
    logs: List[str]
    fallback_mode: bool
    has_ae_component: bool
    optimized_query: str
    language: str


def add_log(current_logs: List[str] | None, message: str) -> List[str]:
    """
    Fügt einen Zeitstempel-versehenen Log-Eintrag hinzu.
    
    Args:
        current_logs: Bestehende Log-Liste (kann None sein)
        message: Log-Nachricht
        
    Returns:
        Aktualisierte Log-Liste mit neuem Eintrag
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    return (current_logs or []) + [f"[{timestamp}] {message}"]


# ==============================================================================
# 3. NODE DEFINITIONS (Graph-Knoten)
# ==============================================================================

def determine_salutation(email_text: str, language: str = "DE") -> str: # <--- language hinzugefügt
    """
    Extrahiert den Namen des ABSENDERS (Signatur) und erstellt eine Anrede.
    Ignoriert explizit den Namen in der Begrüßung (Empfänger).
    """
    # Sprach-Anweisung definieren
    lang_instruction = (
        "Erstelle eine deutsche Anrede (z.B. 'Sehr geehrte Frau Müller,')." 
        if language == "DE" 
        else "Create an English salutation (e.g. 'Dear Ms. Miller,')."
    )

    prompt = f"""
    Du bist ein Assistent, der eine Antwort auf eine eingehende E-Mail verfasst.
    Deine Aufgabe: Erstelle die Anrede für die ANTWORT-Mail an den Verfasser.

    Regeln zur Namensfindung:
    1. Suche den Namen des ABSENDERS. Dieser steht fast immer am ENDE der E-Mail (nach "Viele Grüße", "Mit freundlichen Grüßen", "Signatur").
    2. WICHTIG: Ignoriere Namen, die am ANFANG der E-Mail stehen (z.B. "Hallo Dr. Preuß", "Guten Tag Team"). Das sind die Empfänger, NICHT die Absender.
    3. Beachte das Geschlecht (Dr., Herr, Frau) für die korrekte Anrede (Sehr geehrter Herr..., Sehr geehrte Frau...).
    5. Achte auf akademische Titel (Dr., Prof.) beim Absender.

    Fallback:
    - Wenn KEIN Name in der Signatur/am Ende erkennbar ist, antworte NUR mit: "Sehr geehrte Damen und Herren,"

    Sprach-Anweisung: {lang_instruction}

    E-Mail Text:
    \"\"\"
    {email_text}
    \"\"\"

    Antworte NUR mit der Anrede-Zeile ohne weitere Zeichen.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except:
        return TEMPLATES[language]["salutation_fallback"]

def triage_node(state: AgentState) -> dict:
    """
    Klassifiziert die Anfrage in eine von vier Kategorien.
    """
    question = state["question"]
    # HIER WAR DER FEHLER: Am Ende fehlte die Klammer )
    logs = add_log(state.get("logs"), "TRIAGE: Analysiere Intent & Sprache...") 

    # Prompt für LLM-basierte Klassifikation
    prompt = f"""
    Du bist ein Compliance-Offizier. Kategorisiere die Anfrage in genau EINE Kategorie:
    Aufgabe 1: Bestimme die KATEGORIE der Anfrage basierend auf folgendem Schema:
    1. "ADVERSE_EVENT_ONLY": Nutzer BERICHTET NUR über Vorfall/Symptome, stellt KEINE Frage.
    2. "HYBRID": Nutzer BERICHTET über Vorfall UND stellt eine FRAGE dazu.
    3. "MEDICAL_INFO": Nutzer stellt allgemeine Fragen ohne konkreten Patientenbezug/Vorfall.
    4. "OTHER": Spam, reine Begrüßung.
    Aufgabe 2: Bestimme die SPRACHE der Anfrage (DE oder EN).

    Antworte STRENG im Format: KATEGORIE | SPRACHE
    Beispiel: MEDICAL_INFO | EN
    
    Anfrage: {question}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    
    # Parsing der Antwort "KATEGORIE | SPRACHE"
    try:
        parts = content.split("|")
        category = parts[0].strip()
        language = parts[1].strip().upper()
        if language not in ["DE", "EN"]: language = "DE" # Fallback
    except:
        category = "OTHER"
        language = "DE"

    has_ae = category in ("HYBRID", "ADVERSE_EVENT_ONLY")

    return {
        "category": category,
        "language": language, 
        "has_ae_component": has_ae,
        "logs": add_log(logs, f"TRIAGE: {category} (Sprache: {language})"),
    }


def adverse_event_node(state: AgentState) -> dict:
    """
    Erstellt Standardantwort für reine Nebenwirkungsmeldungen ohne Frage.
    
    Diese Anfragen erfordern keine inhaltliche Recherche, sondern nur 
    eine Bestätigung und Weiterleitung an Pharmacovigilance.
    
    Args:
        state: Aktueller Agent-State
        
    Returns:
        Dict mit draft (Standardtext) und aktualisiertem Log
    """
    question = state["question"]
    lang = state.get("language", "DE") # Sprache laden
    logs = add_log(state.get("logs"), "ADVERSE EVENT: Generiere Bestätigung...")

    txt = TEMPLATES[lang]
    salutation = determine_salutation(question, lang)

    # Text zusammenbauen aus Templates
    response_text = (
        f"{salutation}\n\n"
        f"{txt['header']}\n\n"
        f"{txt['ae_intro']}\n\n" # Hier endet es, da keine Frage beantwortet wird
        f"{txt['footer']}"
    )

    return {
        "draft": response_text,
        "fallback_mode": False,
        "logs": logs,
    }

def retrieve_node(state: AgentState) -> dict:
    # 1. Variablen vorbereiten
    query = state["question"]
    logs = state.get("logs", []) or []
    search_query = query  # <--- HIER: Sicherstellen, dass die Variable immer existiert!

    # 2. Query-Optimierung (deine Logik)
    if len(query) > 30: 
        logs = add_log(logs, "RETRIEVAL: Starte Query-Optimierung...")
        system_prompt = (
            "Extrahiere die medizinischen Kernbegriffe für eine Datenbanksuche. "
            "Entferne Anrede, Gruß und Füllwörter. "
            "Behalte Medikamentennamen und Symptome exakt bei."
        )
        try:
            clean_query = llm.invoke([
                HumanMessage(content=f"{system_prompt}\n\nText: {query}")
            ]).content.strip()
            search_query = clean_query 
            logs = add_log(logs, f"RETRIEVAL: Optimiert zu '{search_query}'")
        except Exception as e:
            logs = add_log(logs, f"Optimierung fehlgeschlagen: {str(e)}")

    # 3. Der eigentliche Datenbank-Abruf (mit Fehler-Diagnose)
    try:
        # Falls hier der Google-Fehler passiert, fangen wir ihn ab:
        docs = retriever.invoke(search_query) 
        logs = add_log(logs, f"RETRIEVAL: {len(docs)} Dokumente gefunden.")
        return {"documents": docs, "logs": logs, "optimized_query": search_query}
    except Exception as e:
        # Dies zeigt dir jetzt den WIRKLICHEN Grund (z.B. API Key falsch, Quote voll, etc.)
        st.error(f"🚨 ECHTER FEHLER VON GOOGLE: {str(e)}")
        # Wir schalten in den Fallback-Modus, damit die App nicht komplett abstürzt
        return {
            "documents": [], 
            "logs": add_log(logs, f"KRITISCHER FEHLER: {str(e)}"), 
            "fallback_mode": True,
            "optimized_query": search_query
        }


def grade_documents_node(state: AgentState) -> dict:
    question = state["question"]
    # Nutze hier auch die optimierte Query, falls die Originalfrage zu "dreckig" ist
    # Das hilft dem Grader oft, den Fokus zu behalten:
    target_query = state.get("optimized_query", question)
    
    documents = state.get("documents", [])
    logs = state.get("logs", []) or []
    filtered_docs = []

    for i, doc in enumerate(documents):
        # Prompt zwingt LLM zu einer Begründung
        prompt = f"""
        Du bist ein strenger Prüfer. 
        Frage: {target_query}
        Dokument-Ausschnitt: {doc.page_content}
        
        Enthält das Dokument Informationen zur Beantwortung der Frage?
        Antworte mit JSON Format: {{"reason": "kurze Begründung", "score": "JA" oder "NEIN"}}
        """
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            
            # Einfaches Parsing (robuster als json.loads bei einfachen LLM Antworten)
            is_relevant = "JA" in content or '"score": "JA"' in content
            
            # Logge die Entscheidung für jedes Dokument
            doc_snippet = doc.page_content[:30].replace("\n", " ")
            logs = add_log(logs, f"GRADING Doc #{i+1} ({doc_snippet}...): {content}")

            if is_relevant:
                filtered_docs.append(doc)
        except Exception as e:
            logs = add_log(logs, f"GRADING Error: {e}")

    if not filtered_docs:
        fallback = True
        context_text = ""
        source_names = []
        logs = add_log(logs, "GRADING: ⚠️ Alle Dokumente abgelehnt -> Fallback.")
    else:
        fallback = False
        context_text = "\n\n".join(d.page_content for d in filtered_docs)
        source_names = sorted({d.metadata.get("source", "Unbekannt") for d in filtered_docs})
        logs = add_log(logs, f"GRADING: {len(filtered_docs)} Dokumente akzeptiert.")

    return {
        "documents": filtered_docs,
        "context": context_text,
        "source_names": source_names,
        "fallback_mode": fallback,
        "logs": logs,
    }


def build_instruction(critique: str | None) -> str:
    """
    Erstellt Prompt-Instruction für Draft-Node.
    
    Die Instruction definiert, dass nur der inhaltliche Body geschrieben 
    werden soll (keine Anrede/Grußformel). Bei vorliegender Kritik wird 
    diese in die Instruction integriert.
    
    Args:
        critique: Feedback vom Critique-Node (None oder String)
        
    Returns:
        Vollständige Instruction für das LLM
    """
    base = (
        "Formuliere NUR den inhaltlichen Antwort-Absatz (Body) auf die Frage. "
        "Schreibe KEINE Anrede ('Sehr geehrte...'), KEINE Einleitung ('Vielen Dank...') "
        "und KEINE Grußformel am Ende. Das übernimmt das System. "
        "Konzentriere dich rein auf die medizinische/sachliche Antwort."
    )
    # Bei vorhandener Kritik: Diese zur Verbesserung hinzufügen
    if critique and critique != "PASS":
        return base + f" Kritik umsetzen: {critique}"
    return base


def draft_node(state: AgentState) -> dict:
    """
    Generiert den E-Mail-Entwurf basierend auf Kontext oder Allgemeinwissen.
    
    Workflow:
    ---------
    1. LLM generiert nur inhaltlichen Body (ohne Anrede/Grußformel)
    2. System fügt Standardelemente hinzu:
       - Header (Anrede + Dank)
       - AE-Block (bei has_ae_component=True)
       - Fallback-Warnung (bei fallback_mode=True)
       - Body (vom LLM generiert)
       - Footer (Grußformel)
    
    Args:
        state: Aktueller Agent-State
        
    Returns:
        Dict mit:
        - draft: Vollständiger E-Mail-Text
        - revision_count: Inkrementierter Zähler
        - logs: Aktualisiertes Log
    """
    question = state["question"]
    context = state.get("context", "")
    critique = state.get("critique", "")
    fallback = state.get("fallback_mode", False)
    has_ae = state.get("has_ae_component", False)
    # Sprache laden (Default DE falls leer)
    lang = state.get("language", "DE") 
    
    logs = add_log(state.get("logs"), f"DRAFT: Erstelle Antwort ({lang})...")
    
    # Templates für die aktuelle Sprache laden
    txt = TEMPLATES[lang]

    # Instruction anpassen für LLM
    instruction = build_instruction(critique)
    lang_prompt = f"Antworte in der Sprache: {lang}. " # <--- WICHTIG für das LLM

    if fallback:
        prompt = f"""
        {instruction} {lang_prompt}
        Du bist Medical Information Manager.
        ACHTUNG: Keine internen Dokumente gefunden. Antworte basierend auf Allgemeinwissen (konservativ).
        Frage: {question}
        """
    else:
        prompt = f"""
        {instruction} {lang_prompt}
        Nutze AUSSCHLIESSLICH den Kontext.
        KONTEXT: {context}
        Frage: {question}
        """

    response = llm.invoke([HumanMessage(content=prompt)])
    body_text = response.content.strip()

    # === ZUSAMMENBAU MIT TEMPLATES ===
    
    # 1. Anrede (Sprache übergeben!)
    salutation = determine_salutation(question, lang)

    header = f"{salutation}\n\n{txt['header']}\n\n"

    ae_block = ""
    if has_ae:
        ae_block = f"{txt['ae_intro']}\n\n{txt['ae_transition']}\n"

    fallback_block = ""
    if fallback:
        fallback_block = f"{txt['fallback']}\n\n"

    footer = f"\n\n{txt['footer']}"

    final_response = header + ae_block + fallback_block + body_text + footer

    return {
        "draft": final_response,
        "revision_count": state.get("revision_count", 0) + 1,
        "logs": logs,
    }


def critique_node(state: AgentState) -> dict:
    """
    Prüft die Qualität des generierten Entwurfs.
    
    Kriterien unterscheiden sich je nach Modus:
    - Fallback: Prüfung auf Vollständigkeit und Flüssigkeit
    - Normal: Prüfung auf Belegbarkeit durch Kontext und Halluzinationen
    
    Args:
        state: Aktueller Agent-State
        
    Returns:
        Dict mit:
        - critique: "PASS" oder detaillierte Fehlerbeschreibung
        - logs: Aktualisiertes Log
    """
    draft = state["draft"]
    question = state["question"]
    fallback = state.get("fallback_mode", False)
    logs = state.get("logs", [])

    # Kriterien je nach Modus
    criteria = (
        "1. Wurde die Frage beantwortet? 2. Klingt der Text flüssig?"
        if fallback
        else "1. Sind alle Aussagen durch den Kontext belegt? 2. Keine Halluzinationen? 3. Werden Nebenwirkungen korrekt an PV verwiesen?"
    )

    # Prompt fordert nun Begründung UND Urteil
    prompt = f"""
    Du bist ein Senior Medical Reviewer. Prüfe den E-Mail Entwurf streng.
    
    Frage: {question}
    Entwurf: {draft}
    Kriterien: {criteria}
    
    Antworte exakt in diesem Format:
    REASONING: [Hier deine detaillierte Begründung, warum gut oder schlecht]
    VERDICT: [PASS oder FAIL: Fehlerbeschreibung]
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # Wir parsen die Antwort (einfaches String-Splitting)
    reasoning = "Keine Begründung generiert."
    verdict = "FAIL"

    if "VERDICT:" in content:
        parts = content.split("VERDICT:")
        reasoning = parts[0].replace("REASONING:", "").strip()
        verdict = parts[1].strip()
    else:
        verdict = content # Fallback, falls Format nicht eingehalten wird

    # 1. Logge die detaillierte Begründung (für den File-Download)
    logs = add_log(logs, f"CRITIQUE DETAIL: {reasoning}")
    
    # 2. Logge das kurze Ergebnis (für den Workflow)
    if "PASS" in verdict:
        logs = add_log(logs, "CRITIQUE RESULT: ✅ PASS")
        final_critique = "PASS"
    else:
        logs = add_log(logs, f"CRITIQUE RESULT: ❌ {verdict}")
        final_critique = verdict

    return {"critique": final_critique, "logs": logs}


# ==============================================================================
# 4. GRAPH CONSTRUCTION (LangGraph State Machine)
# ==============================================================================

# Graph initialisieren
workflow = StateGraph(AgentState)

# Alle Nodes registrieren
workflow.add_node("triage", triage_node)
workflow.add_node("adverse_event", adverse_event_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("draft", draft_node)
workflow.add_node("critique", critique_node)

# Einstiegspunkt definieren
workflow.set_entry_point("triage")


def check_triage(state: AgentState) -> str:
    """
    Routing-Funktion nach Triage-Node.
    
    Entscheidet basierend auf Kategorie, wohin der Flow weitergeht:
    - ADVERSE_EVENT_ONLY → adverse_event (Standardantwort)
    - HYBRID/MEDICAL_INFO → retrieve (RAG-Pipeline)
    - OTHER → end (Abbruch)
    
    Args:
        state: Aktueller Agent-State
        
    Returns:
        String mit Routing-Ziel ("go_ae_only", "go_retrieve", "end")
    """
    cat = state.get("category", "OTHER")
    if cat == "ADVERSE_EVENT_ONLY":
        return "go_ae_only"
    if cat in ("HYBRID", "MEDICAL_INFO"):
        return "go_retrieve"
    if cat == "OTHER":
        return "end"
    return "go_retrieve"  # Fallback


# Conditional Edge nach Triage
workflow.add_conditional_edges(
    "triage",
    check_triage,
    {"end": END, "go_ae_only": "adverse_event", "go_retrieve": "retrieve"},
)

# Lineare Edges für Hauptpfade
workflow.add_edge("adverse_event", END)  # AE-Only → direkt Ende
workflow.add_edge("retrieve", "grade_documents")  # Retrieve → Grading
workflow.add_edge("grade_documents", "draft")  # Grading → Draft
workflow.add_edge("draft", "critique")  # Draft → Critique


def check_critique(state: AgentState) -> str:
    """
    Routing-Funktion nach Critique-Node.
    
    Entscheidet, ob weitere Revision nötig ist oder Workflow beendet wird:
    - PASS oder max. 2 Revisionen erreicht → end
    - Sonst → retry (zurück zu Draft)
    
    Args:
        state: Aktueller Agent-State
        
    Returns:
        String mit Routing-Ziel ("end", "retry")
    """
    if state.get("revision_count", 0) > 2 or state.get("critique") == "PASS":
        return "end"
    return "retry"


# Conditional Edge nach Critique (mit Retry-Loop)
workflow.add_conditional_edges(
    "critique",
    check_critique,
    {"retry": "draft", "end": END},
)

# Graph kompilieren (erstellt ausführbare State Machine)
app = workflow.compile()


# ==============================================================================
# 5. STREAMLIT FRONTEND
# ==============================================================================

# Sidebar: Status und Visualisierung
with st.sidebar:
    st.header("⚙️ System-Status")
    
    # Status-Metriken in schönen Karten
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🗄️ Datenbank", "Aktiv", delta="OK")
    with col2:
        st.metric("🤖 LLM", "gemma-3-27b-it", delta="Online")
    
    st.info(f"📂 DB-Pfad: `{DB_FOLDER}`")
    
    st.markdown("---")
    st.subheader("📊 Workflow-Struktur")
    
    # === HIER BEGINNT DER NEUE TEIL ===
    
    # Graphviz Visualisierung
    workflow_graph = """
    digraph {
        // Layout-Einstellungen
        rankdir=TB;
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, margin=0.2];
        edge [fontsize=9, fontname="Arial", color="#64748b"];

        // --- KNOTEN DEFINITIONEN ---
        START [shape=circle, label="Start", fillcolor="#e2e8f0", width=0.8];
        END [shape=doublecircle, label="Ende", fillcolor="#e2e8f0", width=0.8];

        // Entscheidungsknoten (Raute)
        TRIAGE [shape=diamond, label="Triage\n(Intent)", fillcolor="#bfdbfe", color="#1e3a8a"];
        CRITIQUE [shape=diamond, label="Critique\n(Qualität)", fillcolor="#fef08a", color="#854d0e"];
        GRADE [shape=diamond, label="Grading\n(Relevanz)", fillcolor="#dcfce7", color="#166534"];

        // Aktionsknoten (Box)
        AE_NODE [label="🚑 Adverse Event\n(Meldung an PV)", fillcolor="#fee2e2", color="#991b1b"];
        RETRIEVE [label="🔍 Retrieval\n(DB Suche)", fillcolor="#f1f5f9", color="#475569"];
        DRAFT [label="✍️ Draft\n(Antwort)", fillcolor="#dbeafe", color="#1e40af"];

        // --- KANTEN / LOGIK ---
        START -> TRIAGE;
        
        TRIAGE -> AE_NODE [label="Nebenwirkung", color="#ef4444", fontcolor="#ef4444", penwidth=2];
        TRIAGE -> RETRIEVE [label="Med. Info", color="#22c55e", fontcolor="#15803d", penwidth=2];
        TRIAGE -> END [label="Spam/Other", style="dashed"];

        AE_NODE -> END;

        RETRIEVE -> GRADE;
        GRADE -> DRAFT [label="Docs OK"];
        GRADE -> DRAFT [label="Fallback", style="dashed", fontcolor="#d97706"];

        DRAFT -> CRITIQUE;
        CRITIQUE -> END [label="✅ PASS", color="#22c55e", fontcolor="#15803d"];
        CRITIQUE -> DRAFT [label="❌ RETRY\n(max 2)", color="#eab308", fontcolor="#a16207", style="dashed"];
    }
    """
    
    # Rendern des Graphen
    with st.expander("🔍 Workflow ansehen", expanded=False):
        st.graphviz_chart(workflow_graph, use_container_width=True)
    
    st.caption("🔵 Triage | 🔴 PV-Meldung | 🟢 RAG-Prozess | 🟡 Quality-Check")
    
    # === HIER ENDET DER NEUE TEIL ===

    st.markdown("---")
    st.markdown("### 📖 Kategorien")
    st.markdown("""
    <div style='color: white;'>
    
    - <b>ADVERSE_EVENT</b>: Nebenwirkungsmeldung
    - <b>HYBRID</b>: Meldung + Frage
    - <b>MEDICAL_INFO</b>: Allgemeine Frage
    - <b>OTHER</b>: Sonstige Anfrage
    
    </div>
    """, unsafe_allow_html=True)

# Hauptbereich: Titel und Beschreibung
st.title("🧬 Medical Affairs AI Agent")

st.markdown("""
<div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
            padding: 1.5rem; border-radius: 10px; border-left: 5px solid #3b82f6; margin-bottom: 2rem;'>
    <h3 style='color: #1e3a8a; margin: 0;'>Intelligente Anfragenbearbeitung mit RAG & Compliance</h4>
    <p style='color: #1e40af; margin: 0.5rem 0 0 0;'>
            <strong>Features:</strong> Triage • RAG • AE-Warning • Fallback • Quality-check • Human-in-the-loop • Audit-Logging • Log-Download
    </p>
</div>
""", unsafe_allow_html=True)

# Import (falls noch nicht oben im Skript vorhanden)
import os 

# 1. Wir ermitteln den Ordner, in dem dein Skript (app_v5.4.py) liegt
script_directory = os.path.dirname(os.path.abspath(__file__))

# 2. Wir bauen den vollen Pfad zum Bild
image_path = os.path.join(script_directory, "Der KI-Assistent für Medical Affairs_Infografik.png")

# 3. Einfügen in die App mit Sicherheits-Check
with st.expander("ℹ️ Funktionsweise: Prozess-Grafik anzeigen", expanded=False):
    if os.path.exists(image_path):
        st.image(image_path, 
                 caption="Der Workflow des Agenten im Detail", 
                 use_container_width=True)
    else:
        # Falls es immer noch nicht geht, zeigt uns diese Fehlermeldung, wo Python gesucht hat
        st.error(f"Bild nicht gefunden! Das Skript sucht hier: {image_path}")
# -------------------------


# Eingabebereich mit verbessertem Layout
st.subheader("📝 Ihre Anfrage")
email_input = st.text_area(
    "Geben Sie hier die Anfrage ein:", 
    height=150, 
    value=DEFAULT_QUESTION,
    placeholder="Beschreiben Sie Ihre medizinische Anfrage oder Nebenwirkungsmeldung..."
)

# Button zum Starten des Workflows mit verbessertem Spacing
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    submit_button = st.button("🚀 Anfrage senden", use_container_width=True, type="primary")

if submit_button:
    # Initial State mit allen Pflichtfeldern vorbelegen
    initial_state: AgentState = {
        "question": email_input,
        "revision_count": 0,
        "logs": [],
        "fallback_mode": False,
        "has_ae_component": False,
        "category": "",
        "context": "",
        "documents": [],
        "source_names": [],
        "draft": "",
        "critique": "",
    }
    
    # Workflow ausführen mit Spinner-Anzeige
    with st.spinner("🔄 Agent analysiert und formuliert Antwort..."):
        result = app.invoke(initial_state)
        st.session_state["result"] = result

# ... nach st.session_state["result"] = result ...

if "result" in st.session_state:
    res = st.session_state["result"]
    
    # === NEUER DEBUG BEREICH ===
    st.markdown("---")
    st.subheader("🕵️‍♂️ Debugging Dashboard")
    
    # 1. Query Vergleich
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("**Original Anfrage:**")
        st.info(res.get("question", ""))
    with d_col2:
        st.markdown("**Optimierte DB-Query:**")
        # Rot markieren, wenn sie leer oder seltsam ist
        opt_q = res.get("optimized_query", "N/A")
        if not opt_q or len(opt_q) < 3:
            st.error(f"⚠️ Warnung: '{opt_q}'")
        else:
            st.success(f"'{opt_q}'")

# 2. Grading Entscheidungen visualisieren
    with st.expander("🔍 Detaillierte Filter-Protokolle (Grading)", expanded=True):
        # Wir filtern die Logs nach Grading-Einträgen
        grading_logs = [l for l in res.get("logs", []) if "GRADING" in l]
        
        for log in grading_logs:
            # Bereinigung: Entferne Markdown-Code-Zeichen aus dem Log-Text für die Anzeige
            clean_log = log.replace("```json", "").replace("```", "").replace("{", "(\n  ").replace("}", "\n)").strip()
            
            if "Doc #" in log:
                # Entscheidung visualisieren
                if '"score": "JA"' in log or "JA" in log:
                    # HIER GEÄNDERT: Keine Backticks um {clean_log}
                    st.success(f"✅ **AKZEPTIERT:**\n{clean_log}") 
                else:
                    # HIER GEÄNDERT: Keine Backticks um {clean_log}
                    st.error(f"❌ **ABGELEHNT:**\n{clean_log}")
            else:
                st.caption(log)

# Trennlinie
st.markdown("---")

# Ergebnis-Anzeige (nur wenn vorhanden)
if "result" in st.session_state:
    res: AgentState = st.session_state["result"]
    cat = res.get("category", "OTHER")
    
    # --- FEHLERBEHEBUNG START: Variablen definieren ---
    draft_text = res.get("draft", "")
    context_text = res.get("context", "")
    # --- FEHLERBEHEBUNG ENDE ---

    # === LOG-DATEI STRUKTURIEREN ===
    
    # 1. Metadaten Header
    log_header = f"""
==============================================================================
MEDICAL AFFAIRS AI AGENT - AUDIT LOG
==============================================================================
Date:       {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Category:   {cat}
Result:     {'✅ Sent' if draft_text else '⚠️ No Draft'} 
Revision:   {res.get("revision_count", 0)}
Fallback:   {res.get("fallback_mode")}
------------------------------------------------------------------------------
Original Question:
{res.get("question")}
------------------------------------------------------------------------------
"""

    # 2. System Logs
    formatted_logs = "\n".join(res.get("logs", []))
    log_section = f"\n\n=== SYSTEM LOGS & DECISIONS ===\n{formatted_logs}"

    # 3. Finaler Entwurf
    draft_section = f"\n\n=== FINAL GENERATED DRAFT ===\n{draft_text if draft_text else 'No draft generated.'}"

    # 4. Kontext Audit
    # Falls wir Source-Namen haben, listen wir diese auch auf
    sources_list = "\n- ".join(res.get("source_names", []))
    source_section = f"\nSources used:\n- {sources_list}\n" if res.get("source_names") else ""

    audit_section = f"""
\n\n==============================================================================
APPENDIX: CONTEXT AUDIT (RAG DATA)
==============================================================================
{source_section}
USED TEXT SNIPPETS (Fed to LLM):
--------------------------------
{context_text if context_text else 'No context used (Fallback or no docs found).'}
==============================================================================
"""

    # Alles zusammenfügen
    full_log_text = log_header + log_section + draft_section + audit_section

    # Timestamp für Dateinamen
    tstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(REPORT_FOLDER, f"log_{tstamp}.txt")
    
    # Datei speichern
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(full_log_text)

    # Header mit Kategorie-Badge und Download-Button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("📊 Ergebnisse")
    with col3:
        st.download_button(
            label="📥 Log herunterladen",
            data=full_log_text,
            file_name=os.path.basename(log_path),
            mime="text/plain",
            use_container_width=True
        )

    # Expander für detailliertes System-Log
    with st.expander("📜 Detailliertes System-Log", expanded=False):
        st.markdown("**Workflow-Verlauf mit Timestamps:**")
        for line in res.get("logs", []):
            st.text(line)

    # === ERGEBNIS-DARSTELLUNG ===
    if draft_text:  # Jetzt funktioniert diese Abfrage, da draft_text definiert ist
        # Warnungen je nach Kategorie
        if cat == "ADVERSE_EVENT_ONLY":
            st.warning("⚠️ **REINE NEBENWIRKUNGSMELDUNG** – Standardprozess aktiviert")
        elif cat == "HYBRID":
            st.warning("⚠️ **HYBRID-ANFRAGE** – Nebenwirkung gemeldet + inhaltliche Frage beantwortet")

        # Zwei-Spalten-Layout
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### 📋 Metadaten")
            
            # Fallback-Status oder Quellenangaben
            if res.get("fallback_mode"):
                st.info("ℹ️ **Fallback-Modus**\n\nKeine DB-Dokumente gefunden")
            elif cat != "ADVERSE_EVENT_ONLY":
                st.success("✅ **Quellen genutzt**")
                st.markdown("**Verwendete Dokumente:**")
                # Quelldateien auflisten
                for idx, s in enumerate(res.get("source_names", []), 1):
                    st.caption(f"{idx}. 📄 {os.path.basename(s)}")

                # Audit-Expander
                with st.expander("🔍 Kontext-Audit", expanded=False):
                    st.info("**Verwendete Textauszüge:**")
                    # Hier nutzen wir nun die korrekte Variable context_text
                    st.markdown(f"```text\n{context_text}\n```")
            
            # Statistiken
            st.markdown("---")
            st.metric("🔄 Revisionen", res.get("revision_count", 0))
            st.metric("📄 Dokumente", len(res.get("source_names", [])))
        
        with col1:
            # E-Mail-Vorschau
            st.markdown("### ✉️ Generierter Antwort-Entwurf")
            st.text_area(
                "E-Mail Vorschau", 
                draft_text, 
                height=500,
                label_visibility="collapsed"
            )

            # Sende-Button
            st.markdown("<br>", unsafe_allow_html=True)
            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("✉️ Antwort senden", type="primary", use_container_width=True):
                    st.toast("✅ E-Mail erfolgreich versendet!", icon="✅")
                    st.success(f"✅ Antwort für Ticket **#{tstamp}** wurde verschickt.")
                    st.balloons()
            with btn_col2:
                if st.button("📋 In Zwischenablage kopieren", use_container_width=True):
                    st.toast("📋 Text in Zwischenablage kopiert!", icon="📋")
    else:
        # Kein Draft generiert
        st.info(f"ℹ️ **Kategorie:** {cat} – Keine Antwort generiert.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p style='margin: 0;'>🧬 Medical Affairs AI Agent v5.4 | Powered by LangGraph & Gemma-3-27b-it</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
        © 2025 | Dr. Eike Bent Preuß | Medical Affairs Solutions GmbH
    </p>
</div>
""", unsafe_allow_html=True)









