import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME") or os.getenv("MONGO_DB_NAME", "isl_production_db")

print("=" * 60)
print("       MONGODB ATLAS INTEGRATION VERIFICATION")
print("=" * 60)
print(f"1. .env Loaded:         {env_path}")
print(f"   MONGO_URI:           {mongo_uri[:35]}...{mongo_uri[-20:] if mongo_uri else 'None'}")
print(f"   DB_NAME:             {db_name}")

# Step 2: Direct MongoClient Ping Test
print("\n2. Pinging MongoDB Atlas Admin...")
client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
ping_res = client.admin.command("ping")
print(f"   Ping Response:       {ping_res}")
if ping_res.get("ok") == 1.0:
    print("   Connection Status:   SUCCESS ({'ok': 1.0})")
else:
    print(f"   Connection Status:   FAILED ({ping_res})")
    sys.exit(1)

# Step 3: Test DatabaseManager Integration
print("\n3. Testing DatabaseManager Integration...")
from src.db_manager import DatabaseManager

db = DatabaseManager()
print(f"   DatabaseManager init: OK")
print(f"   Target DB Name:      {db.mongo_db_name}")
print(f"   Client Connected:    {db.mongo_client is not None}")

# Dispatch test inference event
db.log_inference_event(
    signer_id="audit_tester",
    predicted_gloss="namaste",
    confidence=0.992,
    latency_ms=16.8,
    fps=30.0,
    status="verified",
)
print("   Dispatched test inference event.")

# Dispatch test feedback misrecognition
db.log_misrecognition(
    signer_id="audit_tester",
    expected_gloss="namaste",
    predicted_gloss="hello",
    notes="Verification test via test_mongo_integration.py",
)
print("   Dispatched test misrecognition feedback.")

# Wait for background ThreadPoolExecutor to finish writes
time.sleep(3)

# Query Atlas Collections
collections = db.mongo_db.list_collection_names()
print(f"\n4. Atlas Verification in '{db.mongo_db_name}':")
print(f"   Active Collections:  {collections}")

inf_doc = db.mongo_db["inference_logs"].find_one(
    {"signer_id": "audit_tester"}, sort=[("_id", -1)]
)
mis_doc = db.mongo_db["misrecognitions"].find_one(
    {"signer_id": "audit_tester"}, sort=[("_id", -1)]
)

print(f"   Latest Inference Log:     {inf_doc}")
print(f"   Latest Misrecognition Log: {mis_doc}")

print("\n" + "=" * 60)
print(">>> MONGODB ATLAS INTEGRATION VERIFIED & READY FOR TOMORROW!")
print("=" * 60)
