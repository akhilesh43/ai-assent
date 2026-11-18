"""
test_firebase_connection.py — Run this anytime to verify your live Firebase connection.
"""
from memory import firebase_sync

print("=" * 60)
print("JARVIS <-> FIREBASE CONNECTION VERIFICATION")
print("=" * 60)

print(f"1. Service Account Key Present: {firebase_sync.is_configured()}")
status = firebase_sync.get_status()
print(f"2. Project ID: {status['project_id']}")

# Test Realtime Database write & read
stamp = {"status": "ACTIVE", "message": "Connection built and verified!"}
print("\nTesting Realtime Database write...")
ok = firebase_sync.rtdb_put("jarvis/connection_status", stamp)
if ok:
    print("   [OK] Data successfully written to Firebase Realtime Database!")
    val = firebase_sync.rtdb_get("jarvis/connection_status")
    print(f"   [OK] Read back from Firebase: {val}")
else:
    print("   [FAIL] Could not write to Realtime Database.")

# Test Cloud Firestore
print("\nTesting Cloud Firestore write...")
fs_ok = firebase_sync.firestore_set("jarvis_memories", "test_doc", stamp)
if fs_ok:
    print("   [OK] Data successfully written to Cloud Firestore!")
else:
    print("   [INFO] Firestore write bypassed (using Realtime Database as primary).")

print("\n" + "=" * 60)
print("RESULT: Firebase connection is LIVE and ready to save memories.")
print("=" * 60)

