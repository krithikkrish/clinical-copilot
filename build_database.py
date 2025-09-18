import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from fhir.resources.bundle import Bundle
import logging

# --- CONFIGURATION ---
PATIENT_DATA_DIR = os.path.join("data", "patient_data", "fhir")
MEDICAL_KNOWLEDGE_DIR = "data"
DB_PATH = "my_database"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # A powerful and lightweight model
COLLECTION_NAME = "clinical_rag_new"

# Configure logging to see the script's progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_medical_knowledge():
    """Loads text files from the medical knowledge directory."""
    documents = []
    for filename in os.listdir(MEDICAL_KNOWLEDGE_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(MEDICAL_KNOWLEDGE_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Each file is a single "chunk" or document
                    documents.append({"id": filename, "content": content})
            except Exception as e:
                logging.error(f"Error reading medical file {filepath}: {e}")
    logging.info(f"Loaded {len(documents)} medical knowledge documents.")
    return documents

def parse_patient_fhir_bundle(filepath):
    """Parses a complex FHIR JSON file to extract a clean, concise patient summary."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            bundle_data = json.load(f)
        
        bundle = Bundle.parse_obj(bundle_data)
        patient_summary = []
        patient_id = "Unknown"

        # Iterate through all resources in the patient's file
        for entry in bundle.entry:
            resource = entry.resource
            if resource.resource_type == 'Patient':
                patient_id = resource.id
                name = resource.name[0]
                given_name = " ".join(name.given)
                family_name = name.family
                birth_date = resource.birthDate.isoformat()
                gender = resource.gender
                patient_summary.append(f"Patient Record ID: {patient_id}")
                patient_summary.append(f"Name: {given_name} {family_name}, DOB: {birth_date}, Gender: {gender}")

            elif resource.resource_type == 'Condition':
                condition_text = resource.code.text
                onset_date = getattr(getattr(resource, 'onsetDateTime', None), 'isoformat', lambda: 'N/A')()
                patient_summary.append(f"Condition: {condition_text} (Onset: {onset_date})")

            elif resource.resource_type == 'MedicationRequest':
                med_text = getattr(getattr(resource, 'medicationCodeableConcept', None), 'text', 'Unknown Medication')
                patient_summary.append(f"Medication: {med_text}")
            
            elif resource.resource_type == 'Observation':
                if hasattr(resource, 'code') and hasattr(resource.code, 'text'):
                    if resource.code.text in ["Blood Pressure", "Body Mass Index", "Hemoglobin A1c/Hemoglobin.total in Blood"]:
                        obs_text = resource.code.text
                        value_text = "N/A"
                        if hasattr(resource, 'valueQuantity') and resource.valueQuantity:
                            value_text = f"{resource.valueQuantity.value} {resource.valueQuantity.unit}"
                        patient_summary.append(f"Observation: {obs_text} - {value_text}")

        # The final "chunk" is the full summary for one patient
        return {"id": patient_id, "content": "\n".join(patient_summary)}
    
    except Exception as e:
        logging.error(f"Error parsing FHIR file {filepath}: {e}")
        return None

def load_patient_data():
    """Loads and parses all patient FHIR bundles from the directory."""
    documents = []
    if not os.path.exists(PATIENT_DATA_DIR):
        logging.error(f"Patient data directory not found: {PATIENT_DATA_DIR}")
        return []
        
    for filename in os.listdir(PATIENT_DATA_DIR):
        # We only want the main patient record bundle
        if filename.startswith("hospital_information") and filename.endswith(".json"):
            filepath = os.path.join(PATIENT_DATA_DIR, filename)
            patient_doc = parse_patient_fhir_bundle(filepath)
            if patient_doc:
                documents.append(patient_doc)
    logging.info(f"Loaded and parsed {len(documents)} patient records.")
    return documents

def main():
    """Main function to build the vector database."""
    logging.info("--- Starting Knowledge Base Build ---")

    # Load all data sources
    medical_docs = load_medical_knowledge()
    patient_docs = load_patient_data()
    all_docs = medical_docs + patient_docs
    
    if not all_docs:
        logging.warning("No documents found. Exiting.")
        return

    # Initialize the AI model that turns text into vectors (embeddings)
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Initialize ChromaDB, our vector database. It saves data to the 'my_database' folder.
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Create a "collection" which is like a table in a normal database
    logging.info(f"Setting up ChromaDB collection: {COLLECTION_NAME}")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Prepare data for ChromaDB
    ids = [doc['id'] for doc in all_docs]
    contents = [doc['content'] for doc in all_docs]
    
    # Generate embeddings for all documents. This is the core AI step.
    logging.info("Generating embeddings for all documents... (This may take a moment)")
    embeddings = model.encode(contents, show_progress_bar=True).tolist()

    # Add the documents and their embeddings to the database
    logging.info("Adding documents to the vector database...")
    collection.add(
        embeddings=embeddings,
        documents=contents,
        ids=ids
    )

    logging.info("--- Knowledge Base Build Complete ---")
    logging.info(f"Total documents indexed: {collection.count()}")
    logging.info(f"Database saved in: '{DB_PATH}'")

if __name__ == "__main__":
    main()

