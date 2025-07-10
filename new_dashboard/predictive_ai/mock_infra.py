# Mock infrastructure data for threat prediction
mock_infrastructure = {
    "servers": [
        {
            "hostname": "web-server-01",
            "type": "web_server",
            "os": "Ubuntu 20.04",
            "services": ["nginx", "php-fpm", "mysql"],
            "network": "10.0.1.0/24",
            "exposed_ports": [80, 443, 22]
        },
        {
            "hostname": "web-server-02", 
            "type": "web_server",
            "os": "Ubuntu 20.04",
            "services": ["apache2", "php", "mysql"],
            "network": "10.0.1.0/24",
            "exposed_ports": [80, 443, 22]
        },
        {
            "hostname": "db-server-01",
            "type": "database_server",
            "os": "CentOS 8",
            "services": ["mysql", "redis"],
            "network": "10.0.2.0/24",
            "exposed_ports": [3306, 6379, 22]
        },
        {
            "hostname": "db-server-02",
            "type": "database_server", 
            "os": "CentOS 8",
            "services": ["postgresql", "redis"],
            "network": "10.0.2.0/24",
            "exposed_ports": [5432, 6379, 22]
        },
        {
            "hostname": "file-server-01",
            "type": "file_server",
            "os": "CentOS 7",
            "services": ["samba", "nfs", "ftp"],
            "network": "10.0.3.0/24",
            "exposed_ports": [21, 22, 139, 445, 2049]
        },
        {
            "hostname": "mail-server-01",
            "type": "mail_server", 
            "os": "Ubuntu 22.04",
            "services": ["postfix", "dovecot", "spamassassin"],
            "network": "10.0.4.0/24",
            "exposed_ports": [25, 110, 143, 993, 995]
        },
        {
            "hostname": "backup-server-01",
            "type": "backup_server",
            "os": "Ubuntu 20.04", 
            "services": ["rsync", "bacula"],
            "network": "10.0.5.0/24",
            "exposed_ports": [22, 9101, 9102, 9103]
        },
        {
            "hostname": "backup-server-02",
            "type": "backup_server",
            "os": "CentOS 8",
            "services": ["rsync", "duplicity"],
            "network": "10.0.5.0/24", 
            "exposed_ports": [22, 873]
        }
    ],
    "network_devices": [
        {
            "hostname": "firewall-01",
            "type": "firewall",
            "model": "pfSense",
            "network": "10.0.0.1/16",
            "services": ["firewall", "vpn", "dhcp"]
        },
        {
            "hostname": "router-01", 
            "type": "router",
            "model": "Cisco ISR 4331",
            "network": "192.168.1.1/24",
            "services": ["routing", "nat"]
        },
        {
            "hostname": "switch-01",
            "type": "switch", 
            "model": "Cisco Catalyst 2960",
            "network": "10.0.0.0/16",
            "services": ["switching", "vlan"]
        }
    ],
    "endpoints": [
        {
            "hostname": "workstation-01",
            "type": "workstation",
            "os": "Windows 10",
            "services": ["rdp", "smb"],
            "network": "10.0.10.0/24",
            "exposed_ports": [3389, 445]
        }
    ],
    "security_tools": [
        {
            "name": "SentrySense",
            "type": "anomaly_detection",
            "components": ["GNN", "CVE_monitor", "dashboard"]
        }
    ]
}
