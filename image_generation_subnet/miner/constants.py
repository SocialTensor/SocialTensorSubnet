NGINX_CONF = """
worker_processes 1;
events {
    worker_connections 1024;
}

http {
    ignore_invalid_headers off;
    client_max_body_size 0;
    proxy_intercept_errors on;
    server {
        listen {{external_axon_port}};  # Port to listen on

        # Whitelist IP addresses
        # allow 123.45.67.89;  # Replace with your allowed IPs
        # allow 98.76.54.32;   # You can add multiple allowed IPs
        {{whitelist}}
        deny all;            # Deny all other IPs
        access_log /var/log/nginx/access.log proxylog;
        proxy_http_version 1.1;

        location / {
            proxy_pass http://127.0.0.1:{{internal_axon_port}};
        }
    }
}

"""
