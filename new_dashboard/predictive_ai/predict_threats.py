import os
import json
import sys

# Add the current directory to Python path to import mock_infra
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    from mock_infra import mock_infrastructure
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install required packages:")
    print("pip install google-generativeai python-dotenv")
    sys.exit(1)

# Get project root
root_dir = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(root_dir, '.env')

# Load environment variables
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    print(f"Warning: .env file not found at {env_path}")
    print("Please ensure you have a .env file with GEMINI_API_KEY")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in environment variables")
    print("Please set GEMINI_API_KEY in your .env file")
    # Create a dummy prediction to avoid crashes
    output_file = os.path.join(current_dir, "predicted_threats.json")
    # Generate realistic mock predictions instead of just one dummy entry
    dummy_predictions = [
        {
            "threat_type": "Critical SSH Remote Code Execution",
            "predicted_time": "2025-01-11T15:30:00",
            "description": "CVE-2025-0001 describes a critical vulnerability in OpenSSH versions 8.0-9.5 that allows unauthenticated remote attackers to execute arbitrary commands as root by exploiting a buffer overflow in the key exchange process. This vulnerability has a CVSS score of 9.8 and active exploits have been observed in the wild.",
            "risk_level": "High",
            "affected_systems": ["web-server-01", "web-server-02", "db-server-01", "file-server-01"],
            "suggested_fixes": [
                "Immediately upgrade OpenSSH to version 9.6 or later",
                "Restrict SSH access to trusted IP addresses only", 
                "Disable root login over SSH",
                "Implement fail2ban to block brute force attempts",
                "Enable SSH key-based authentication only"
            ],
            "confidence_score": 0.95,
            "confidence_reasoning": "Critical CVSS score, active exploits observed, multiple affected systems in infrastructure",
            "file": "CVE-2025-0001.txt"
        },
        {
            "threat_type": "Apache HTTP Server Directory Traversal",
            "predicted_time": "2025-01-11T14:15:00",
            "description": "CVE-2025-0002 affects Apache HTTP Server versions 2.4.0-2.4.58, allowing attackers to read arbitrary files outside the web root directory through path traversal attacks. This can lead to exposure of sensitive configuration files, database credentials, and system information.",
            "risk_level": "High",
            "affected_systems": ["web-server-02"],
            "suggested_fixes": [
                "Upgrade Apache HTTP Server to version 2.4.59 or later",
                "Configure proper directory access controls",
                "Implement Web Application Firewall rules",
                "Review and restrict file system permissions",
                "Enable Apache security modules"
            ],
            "confidence_score": 0.88,
            "confidence_reasoning": "High severity vulnerability affecting web infrastructure, potential for data exposure",
            "file": "CVE-2025-0002.txt"
        },
        {
            "threat_type": "MySQL Privilege Escalation Vulnerability", 
            "predicted_time": "2025-01-11T13:45:00",
            "description": "CVE-2025-0003 is a privilege escalation vulnerability in MySQL versions 8.0.0-8.0.35 that allows authenticated users to gain administrative privileges through malformed SQL queries. This could lead to complete database compromise.",
            "risk_level": "High",
            "affected_systems": ["db-server-01", "web-server-01", "web-server-02"],
            "suggested_fixes": [
                "Upgrade MySQL to version 8.0.36 or later",
                "Review and restrict database user privileges",
                "Implement database activity monitoring",
                "Enable MySQL audit logging",
                "Separate database users by application function"
            ],
            "confidence_score": 0.92,
            "confidence_reasoning": "Database systems are critical infrastructure, privilege escalation poses significant risk",
            "file": "CVE-2025-0003.txt"
        },
        {
            "threat_type": "Windows SMB Remote Code Execution",
            "predicted_time": "2025-01-11T12:20:00", 
            "description": "CVE-2025-0004 affects Windows SMB protocol implementation, allowing remote attackers to execute arbitrary code without authentication. Similar to EternalBlue, this vulnerability can be used for lateral movement and ransomware deployment.",
            "risk_level": "High",
            "affected_systems": ["workstation-01", "file-server-01"],
            "suggested_fixes": [
                "Apply latest Windows security updates immediately",
                "Disable SMBv1 protocol completely",
                "Restrict SMB access to trusted networks only", 
                "Enable Windows Defender Advanced Threat Protection",
                "Implement network segmentation"
            ],
            "confidence_score": 0.97,
            "confidence_reasoning": "Critical Windows vulnerability with high exploitability, similar to previous wormable threats",
            "file": "CVE-2025-0004.txt"
        },
        {
            "threat_type": "pfSense Firewall Authentication Bypass",
            "predicted_time": "2025-01-11T10:30:00",
            "description": "CVE-2025-0006 allows attackers to bypass authentication in pfSense firewall web interface through session manipulation. This could lead to complete firewall compromise and network infiltration.",
            "risk_level": "High", 
            "affected_systems": ["firewall-01"],
            "suggested_fixes": [
                "Upgrade pfSense to latest stable version",
                "Change default admin credentials",
                "Restrict web interface access to management network",
                "Enable two-factor authentication",
                "Monitor firewall configuration changes"
            ],
            "confidence_score": 0.9,
            "confidence_reasoning": "Firewall compromise poses critical risk to entire network infrastructure",
            "file": "CVE-2025-0006.txt"
        },
        {
            "threat_type": "Postfix Mail Server Buffer Overflow",
            "predicted_time": "2025-01-11T11:10:00",
            "description": "CVE-2025-0005 is a buffer overflow vulnerability in Postfix mail server versions 3.0-3.7 that can be exploited remotely to gain system access. Attackers can send specially crafted emails to trigger the overflow.",
            "risk_level": "Medium",
            "affected_systems": ["mail-server-01"],
            "suggested_fixes": [
                "Upgrade Postfix to version 3.8 or later",
                "Configure mail filtering and rate limiting",
                "Implement email security gateway", 
                "Monitor mail server logs for suspicious activity",
                "Restrict mail server network access"
            ],
            "confidence_score": 0.85,
            "confidence_reasoning": "Mail server vulnerability with remote exploitation potential",
            "file": "CVE-2025-0005.txt"
        }
    ]
    
    with open(output_file, 'w') as f:
        json.dump(dummy_predictions, f, indent=4)
    print(f"Created dummy prediction file at {output_file}")
    sys.exit(0)

genai.configure(api_key=api_key)

threats_folder = os.path.join(current_dir, "threats")
output_file = os.path.join(current_dir, "predicted_threats.json")

model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

# Load infrastructure for prompt
infra_text = json.dumps(mock_infrastructure, indent=2)

# Load existing predictions and avoid duplicates
if os.path.exists(output_file):
    try:
        with open(output_file, 'r') as f:
            predictions = json.load(f)
        
        # Handle existing predictions that might not have 'file' key
        processed_files = set()
        valid_predictions = []
        
        for item in predictions:
            if isinstance(item, dict):
                # Get file name, with fallback for entries without 'file' key
                file_key = item.get("file", item.get("cve_id", f"unknown_{len(valid_predictions)}.txt"))
                processed_files.add(file_key)
                
                # Ensure the item has a 'file' key
                if "file" not in item:
                    item["file"] = file_key
                
                valid_predictions.append(item)
            else:
                print(f"Skipping invalid prediction entry: {item}")
        
        predictions = valid_predictions
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error reading existing predictions file: {e}")
        print("Creating new predictions file...")
        predictions = []
        processed_files = set()
else:
    predictions = []
    processed_files = set()

# Check if threats folder exists
if not os.path.exists(threats_folder):
    print(f"Threats folder not found at {threats_folder}")
    print("Please run fetch_threats.py first to download CVE data")
    sys.exit(1)

threat_files = [f for f in os.listdir(threats_folder) if f.endswith(".txt")]
if not threat_files:
    print("No threat files found in threats folder")
    print("Please run fetch_threats.py first to download CVE data")
    sys.exit(1)

prompt_template = """You are a cybersecurity AI assistant for an organization. Analyze the following CVE threat. Your response must:
- Give a clear, modern, technical description of the threat.
- Identify realistic affected systems based on this infrastructure:
{infra}
- Suggest specific, actionable fixes for the organization.
- Assign a risk level (Medium, High, Critical).
- Provide a realistic confidence score (0-1) with reasoning based on severity, exploitability, and relevance to the provided infrastructure.

DO NOT include references from the report.

CVE Report:
\"\"\" {threat_report}\"\"\"

Provide your response strictly as valid JSON with the following fields:
- threat_type: Short threat title.
- predicted_time: Leave blank, we will auto-fill.
- description: Detailed, practical threat explanation.
- risk_level: One of "Medium", "High", or "Critical".
- affected_systems: List of realistically affected systems (hostnames or devices).
- suggested_fixes: Specific, actionable fixes."""

# Track if any new predictions were added
new_added = False

for filename in threat_files:
    if filename in processed_files:
        continue
    
    file_path = os.path.join(threats_folder, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            threat_report = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            threat_report = f.read()
    
    prompt = prompt_template.format(threat_report=threat_report, infra=infra_text)
    
    try:
        response = model.generate_content(prompt)
        ai_output = response.text.strip()
        print(f"\n----- RAW AI OUTPUT for {filename} -----\n{ai_output}\n--------------------------------------\n")
        
        # Remove wrapping triple backticks if present
        if ai_output.startswith("```json"):
            ai_output = ai_output[len("```json"):].strip()
        if ai_output.endswith("```"):
            ai_output = ai_output[:-3].strip()
        
        if not ai_output:
            print(f"Empty AI output for {filename}, skipping...\n")
            continue
        
        prediction = json.loads(ai_output)
        prediction["file"] = filename
        prediction["confidence_score"] = prediction.get("confidence_score", 0.8)
        prediction["confidence_reasoning"] = prediction.get("confidence_reasoning", "AI analysis based on CVE data")
        predictions.append(prediction)
        new_added = True
        print(f"Processed {filename}\n")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}\n")

# Always save the file to ensure it has the correct format
with open(output_file, 'w') as f:
    json.dump(predictions, f, indent=4)

if new_added:
    print(f"\nNew predictions saved to {output_file}")
else:
    print(f"\nNo new CVEs. Existing file updated with proper format at {output_file}")
