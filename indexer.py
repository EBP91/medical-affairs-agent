import os
import shutil
import time  # WICHTIG für die Pausen
from dotenv import load_dotenv

# LangChain Importe
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Umgebungsvariablen laden
load_dotenv()

# Konfiguration
DATA_FOLDER = "data"
DB_FOLDER = "chroma_db"

def main():
    # Check: API Key vorhanden?
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Fehler: GOOGLE_API_KEY nicht gefunden. Bitte in .env eintragen.")
        return

    # Alte DB löschen für sauberen Neustart
    if os.path.exists(DB_FOLDER):
        print(f"🗑️  Lösche alte Datenbank '{DB_FOLDER}'...")
        shutil.rmtree(DB_FOLDER)

    documents = []
    print(f"📂 Durchsuche Ordner '{DATA_FOLDER}' nach PDFs...")
    
    # PDFs laden
    if os.path.exists(DATA_FOLDER):
        for filename in os.listdir(DATA_FOLDER):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(DATA_FOLDER, filename)
                print(f"   📄 Lade: {filename}")
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"   ⚠️ Fehler bei {filename}: {e}")

    if not documents:
        print("❌ Keine Dokumente geladen.")
        return

    print(f"✅ {len(documents)} Seiten geladen.")

    # Text splitten
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Text in {len(chunks)} Chunks zerteilt.")

    # --- HIER IST DIE WICHTIGE ÄNDERUNG ---
    print("💾 Starte 'Slow & Safe' Upload zu ChromaDB...")
    
    # 1. Embedding Funktion vorbereiten
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", # Dein korrektes Modell
        google_api_key=api_key
    )

    # 2. Leere Datenbank initialisieren (NICHT from_documents benutzen!)
    vector_store = Chroma(
        embedding_function=embedding_function,
        persist_directory=DB_FOLDER
    )

    # 3. Manuelle Batch-Schleife mit PAUSEN
    batch_size = 5  # Nur 5 auf einmal
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, batch_size):
        # Nimm 5 Stück
        batch = chunks[i : i + batch_size]
        
        print(f"   ⏳ Verarbeite {i+1} bis {min(i+batch_size, total_chunks)} von {total_chunks}...", end=" ")
        
        try:
            # Senden
            vector_store.add_documents(batch)
            print("✅ OK. Warte 5s...")
            # WICHTIG: Die Bremse!
            time.sleep(5) 
        except Exception as e:
            print(f"\n   ❌ Fehler im Batch: {e}")
            print("   ⚠️ Warte 60 Sekunden wegen Rate Limit...")
            time.sleep(60) # Lange Pause bei Fehler
            try:
                vector_store.add_documents(batch) # Nochmal versuchen
                print("   ✅ Retry erfolgreich.")
            except:
                print("   ❌ Batch endgültig übersprungen.")

    print(f"\n🎉 FERTIG! Datenbank in '{DB_FOLDER}' erstellt.")
    print("👉 Bitte jetzt den Ordner 'chroma_db' auf GitHub hochladen/pushen!")

if __name__ == "__main__":
    main()