#!/usr/bin/env python3
"""
Simplified threat fetching script for SentrySense
This version doesn't use emoji characters and generates mock threat data
"""

import json
import random
import time
from datetime import datetime
from pathlib import Path
import os
import requests

def generate_threat():
    """Generate a mock threat prediction"""
    threat_types = [
        "SQL Injection Vulnerability",
        "Cross-Site Scripting (XSS)",
        "Buffer Overflow Attack",
        "Privilege Escalation",
        "Remote Code Execution",
        "Denial of Service (DoS)",
        "Man-in-the-Middle Attack",
        "Phishing Campaign",
        "Malware Infection",
        "Data Breach Attempt"
    ]
    
    systems = [
        "web-server-01", "web-server-02", "db-server-01", "db-server-02",
        "file-server-01", "mail-server-01", "backup-server-01", "backup-server-02",
        "firewall-01", "router-01", "switch-01", "workstation-01"
    ]
    
    risk_levels = ["Low", "Medium", "High"]
    
    fixes_db = {
        "SQL Injection Vulnerability": [
            "Update web application framework",
            "Implement parameterized queries",
            "Enable input validation",
            "Deploy Web Application Firewall",
            "Conduct security code review"
        ],
        "Cross-Site Scripting (XSS)": [
            "Sanitize user inputs",
            "Implement Content Security Policy",
            "Update web browser policies",
            "Enable XSS protection headers",
            "Validate all output encoding"
        ],
        "Buffer Overflow Attack": [
            "Apply security patches",
            "Enable stack protection",
            "Update vulnerable software",
            "Implement address space layout randomization",
            "Use memory-safe programming languages"
        ]
    }
    
    threat_type = random.choice(threat_types)
    cve_id = f"CVE-2024-{random.randint(1000, 9999)}"
    
    return {
        "threat_type": threat_type,
        "description": f"{cve_id}: {threat_type} detected in network infrastructure. "
                      f"This vulnerability could allow attackers to compromise system security "
                      f"and gain unauthorized access to sensitive data.",
        "risk_level": random.choice(risk_levels),
        "affected_systems": random.sample(systems, random.randint(1, 3)),
        "suggested_fixes": fixes_db.get(threat_type, [
            "Apply latest security patches",
            "Update system software",
            "Review security configurations",
            "Monitor system logs",
            "Implement network segmentation"
        ]),
        "confidence_score": round(random.uniform(0.75, 0.98), 2),
        "cve_id": cve_id,
        "severity_score": round(random.uniform(3.0, 9.5), 1),
        "discovery_date": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }

def main():
    """Main threat fetching function"""
    # Ensure directory exists
    os.makedirs("predictive_ai", exist_ok=True)
    threat_file = Path("predictive_ai/predicted_threats.json")
    
    print("Starting SentrySense Threat Prediction...")
    print(f"Saving to: {threat_file}")
    
    # Load existing threats or create empty list
    if threat_file.exists():
        try:
            with open(threat_file, 'r') as f:
                threats = json.load(f)
                if not isinstance(threats, list):
                    print("Invalid threats file format, creating new file")
                    threats = []
        except (json.JSONDecodeError, UnicodeDecodeError):
            print("Error reading threats file, creating new file")
            threats = []
    else:
        threats = []
    
    # Generate new threat
    if random.random() < 0.7:  # 70% chance of new threat
        new_threat = generate_threat()
        threats.append(new_threat)
        
        # Keep only last 10 threats to prevent file from growing too large
        threats = threats[-10:]
        
        # Save updated threats
        with open(threat_file, 'w') as f:
            json.dump(threats, f, indent=2)
        
        print(f"New threat predicted: {new_threat['threat_type']}")
        print(f"Risk Level: {new_threat['risk_level']}")
        print(f"CVE ID: {new_threat['cve_id']}")
        print(f"Affected Systems: {', '.join(new_threat['affected_systems'])}")
    else:
        print("No new threats predicted in current cycle")
    
    # Create folder to save threat reports inside predictive_ai/threats
    current_dir = os.path.dirname(__file__)  # Gets path to predictive_ai
    output_folder = os.path.join(current_dir, "threats")
    os.makedirs(output_folder, exist_ok=True)

    # Define API URL - Limit to recent 5 CVEs (no severity filter to include both HIGH & MEDIUM)
    url = "https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage=5"

    # Add headers to avoid 404 (pretend to be a browser)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SentrySense/1.0)"
    }

    # Fetch data
    response = requests.get(url, headers=headers)

    # Check response
    if response.status_code != 200:
        print(f"Error fetching data. Status code: {response.status_code}")
        exit()

    data = response.json()

    count = 0  # To track saved files

    # Loop through vulnerabilities
    for item in data.get('vulnerabilities', []):
        cve = item.get('cve', {})
        cve_id = cve.get('id', 'N/A')
        
        # Skip if file already exists
        file_path = os.path.join(output_folder, f"{cve_id}.txt")
        if os.path.exists(file_path):
            print(f"{cve_id} already exists. Skipping...")
            continue
        
        # Extract English description
        descriptions = cve.get('descriptions', [])
        description = next((d['value'] for d in descriptions if d['lang'] == 'en'), 'No description available.')
        
        # Extract severity and score
        metrics = cve.get('metrics', {})
        severity = 'Unknown'
        score = 'Unknown'
        
        if 'cvssMetricV2' in metrics:
            metric = metrics['cvssMetricV2'][0]
            severity = metric.get('baseSeverity', 'Unknown')
            score = metric.get('cvssData', {}).get('baseScore', 'Unknown')
        
        # Extract published date
        published_date = cve.get('published', 'Unknown')
        
        # Filter for HIGH and MEDIUM severity only
        if severity in ["HIGH", "MEDIUM"]:
            with open(file_path, 'w') as f:
                f.write(f"CVE ID: {cve_id}\n")
                f.write(f"Published Date: {published_date}\n")
                f.write(f"Description: {description}\n")
                f.write(f"Severity: {severity} (Score: {score})\n")
                
                # Add reference links
                references = cve.get('references', [])
                if references:
                    f.write("References:\n")
                    for ref in references:
                        f.write(f" - {ref.get('url', '')}\n")
            
            print(f"Saved {cve_id}.txt ({severity})")
            count += 1

    print(f"\nTotal NEW HIGH/MEDIUM severity threats saved: {count}")
    
    print("Threat prediction completed successfully")

if __name__ == "__main__":
    main()
