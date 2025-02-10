import multiprocessing
import os

# Server socket configuration
bind = os.getenv('BIND', '0.0.0.0:8000')
backlog = 2048

# Worker processes
worker_class = 'gthread'  # Use threads for async operations
workers = 2  # Reduced number of workers to manage memory
threads = 4  # Number of threads per worker
worker_connections = 400
max_requests = 200  # Restart workers after this many requests
max_requests_jitter = 50  # Add randomness to max_requests

# Timeout configuration
timeout = 120  # Increased timeout for long-running tasks
graceful_timeout = 30
keepalive = 5

# Process naming
proc_name = 'picfinder'
pythonpath = '.'

# SSL configuration (if needed)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'

# Logging
accesslog = '-'  # stdout
errorlog = '-'   # stderr
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Development settings
reload = os.getenv('FLASK_ENV', 'production') == 'development'
reload_extra_files = []

# Security settings
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Server mechanics
daemon = False
raw_env = [
    'FLASK_APP=app.py',
    'FLASK_ENV=' + os.getenv('FLASK_ENV', 'production')
]
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Memory optimization
max_requests_jitter = 50
worker_tmp_dir = '/dev/shm'  # Use RAM for temporary files
post_fork = None
post_worker_init = None

def post_fork(server, worker):
    """
    Called just after a worker has been forked.
    """
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def worker_int(worker):
    """
    Called just after a worker exited on SIGINT or SIGQUIT.
    """
    worker.log.info("Worker received INT or QUIT signal")

def worker_abort(worker):
    """
    Called when a worker receives SIGABRT signal.
    """
    worker.log.info("Worker received ABORT signal")

def pre_exec(server):
    """
    Called just before a new master process is forked.
    """
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    """
    Called just after the server is started.
    """
    server.log.info("Server is ready. Spawning workers")

def worker_exit(server, worker):
    """
    Called just after a worker has been exited, in the worker process.
    """
    server.log.info("Worker exited (%s)", worker.pid)

# SSL Configuration
keyfile = os.getenv('SSL_KEYFILE', None)
certfile = os.getenv('SSL_CERTFILE', None)
ssl_version = os.getenv('SSL_VERSION', 'TLS')
cert_reqs = os.getenv('SSL_CERT_REQS', 'REQUIRED')
ca_certs = os.getenv('SSL_CA_CERTS', None)
suppress_ragged_eofs = True

# Error handling
capture_output = True
enable_stdio_inheritance = False

# Performance tuning
worker_connections = int(os.getenv('GUNICORN_WORKER_CONNECTIONS', '400'))
max_requests = int(os.getenv('GUNICORN_MAX_REQUESTS', '200'))
timeout = int(os.getenv('GUNICORN_TIMEOUT', '120'))
keepalive = int(os.getenv('GUNICORN_KEEPALIVE', '5'))
graceful_timeout = int(os.getenv('GUNICORN_GRACEFUL_TIMEOUT', '30'))

# Memory management
def pre_request(worker, req):
    """
    Called just before a worker processes the request.
    """
    import gc
    gc.collect()

def post_request(worker, req, environ, resp):
    """
    Called after a worker processes the request.
    """
    import gc
    gc.collect()

# Custom settings for handling large files
# Increase if handling large file uploads
limit_request_line = int(os.getenv('GUNICORN_LIMIT_REQUEST_LINE', '4096'))
limit_request_fields = int(os.getenv('GUNICORN_LIMIT_REQUEST_FIELDS', '100'))
limit_request_field_size = int(os.getenv('GUNICORN_LIMIT_REQUEST_FIELD_SIZE', '8190'))

# Statsd settings (if using statsd for monitoring)
statsd_host = os.getenv('STATSD_HOST', None)
statsd_prefix = os.getenv('STATSD_PREFIX', 'picfinder')

# Custom error log settings
errorlog = os.getenv('GUNICORN_ERROR_LOG', '-')
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')
accesslog = os.getenv('GUNICORN_ACCESS_LOG', '-')
access_log_format = os.getenv(
    'GUNICORN_ACCESS_LOG_FORMAT',
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'
)

# Worker class configuration
worker_class = os.getenv('GUNICORN_WORKER_CLASS', 'gthread')
threads = int(os.getenv('GUNICORN_THREADS', '4'))

# Calculate workers based on CPU cores but limit to prevent memory issues
def get_workers():
    cores = multiprocessing.cpu_count()
    # Use 2 workers maximum to prevent memory issues
    return min(cores, 2)

workers = int(os.getenv('GUNICORN_WORKERS', get_workers()))
