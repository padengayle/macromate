import sqlite3
import json
from mcp.server.fastmcp import FastMCP

# 1. Initialize the MCP Server
mcp = FastMCP("Macromate Health Context")

DB_PATH = "health_profiles.db"

def setup_database():
    """Spins up a local SQLite database with mock enterprise health data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the secure schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            target_daily_calories INTEGER,
            medical_conditions TEXT,
            allergies TEXT
        )
    ''')
    
    # Reset and insert mock data (idempotent for testing)
    cursor.execute('DELETE FROM users')
    mock_data = [
        ("user_123", 2000, "Type 2 Diabetes", "None"),
        ("user_456", 2500, "None", "Peanut, Gluten (Celiac)"),
        ("user_789", 1800, "Hypertension", "Dairy")
    ]
    
    cursor.executemany('''
        INSERT INTO users (user_id, target_daily_calories, medical_conditions, allergies)
        VALUES (?, ?, ?, ?)
    ''', mock_data)
    
    conn.commit()
    conn.close()

# 3. Define the Secure Tool Boundary
@mcp.tool()
def get_user_health_context(user_id: str) -> str:
    """
    Securely retrieves a user's health profile, target calories, and allergies.
    The LLM calls this tool; it NEVER executes raw SQL.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Parameterized query enforces security (prevents LLM prompt-injection to SQL)
    cursor.execute('''
        SELECT target_daily_calories, medical_conditions, allergies 
        FROM users WHERE user_id = ?
    ''', (user_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        profile = {
            "user_id": user_id,
            "target_calories": row[0],
            "medical_conditions": row[1],
            "allergies": row[2]
        }
        return json.dumps(profile, indent=2)
    else:
        return json.dumps({"error": f"No health profile found for {user_id}"})

if __name__ == "__main__":
    print("Initializing Macromate Database...")
    setup_database()
    print("Starting secure MCP server on stdio...")
    # Runs the server using standard I/O, the default for MCP communication
    mcp.run(transport='stdio')
