nginx_conf = """
worker_processes 1;
events {
    worker_connections 1024;
}

http {
    server {
        listen {external_axon_port};  # Port to listen on

        # Whitelist IP addresses
        # allow 123.45.67.89;  # Replace with your allowed IPs
        # allow 98.76.54.32;   # You can add multiple allowed IPs
        {whitelist}
        deny all;            # Deny all other IPs

        location / {
            proxy_pass http://127.0.0.1:{internal_axon_port};
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}

"""
