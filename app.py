# app.py
from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import boto3
import os
import io
import logging
import json
from PIL import Image
import concurrent.futures
import threading
from queue import Queue
import numpy as np
from functools import lru_cache
import time
import math
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, 
     resources={
         r"/*": {
             "origins": [
                 "https://picfinder-develop.web.app",
                 "http://localhost:3000"
             ],
             "methods": ["GET", "POST", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization"],
             "expose_headers": ["Content-Type", "Authorization"],
             "supports_credentials": True,
             "send_wildcard": False,
             "max_age": 86400
         }
     },
     intercept_exceptions=True)

# Setup logging with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.readonly']
MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15MB
REKOGNITION_MAX_SIZE = 5 * 1024 * 1024  # 5MB AWS limit
BATCH_SIZE = 5  # Size of each batch
NUM_PARALLEL_BATCHES = 4  # Number of batches to process in parallel
CACHE_TTL = 3600

# Credentials configuration
CREDENTIALS = {
    "type": "service_account",
    "project_id": os.getenv('GOOGLE_PROJECT_ID'),
    "private_key_id": os.getenv('GOOGLE_PRIVATE_KEY_ID'),
    "private_key": os.getenv('GOOGLE_PRIVATE_KEY').replace('\\n', '\n'),
    "client_email": os.getenv('GOOGLE_CLIENT_EMAIL'),
    "client_id": os.getenv('GOOGLE_CLIENT_ID'),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.getenv('GOOGLE_CLIENT_X509_CERT_URL'),
    "universe_domain": "googleapis.com"
}

# AWS credentials
AWS_CREDENTIALS = {
    "aws_access_key_id": os.getenv('AWS_ACCESS_KEY_ID'),
    "aws_secret_access_key": os.getenv('AWS_SECRET_ACCESS_KEY'),
    "region": os.getenv('AWS_REGION', 'us-east-1')
}

# Initialize cache
class ImageCache:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < CACHE_TTL:
                    return data
                else:
                    del self.cache[key]
            return None

    def set(self, key, value):
        with self.lock:
            self.cache[key] = (value, time.time())

image_cache = ImageCache()

# Thread-local storage
thread_local = threading.local()

class OptimizedImageProcessor:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.process_queue = Queue()
        self.results = []
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.processed_hashes = set()
        
    @staticmethod
    def optimize_image(img_bytes, target_size=(800, 800), quality=85):
        try:
            img = Image.open(io.BytesIO(img_bytes))
            
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            
            aspect = img.size[0] / img.size[1]
            if aspect > 1:
                new_size = (target_size[0], int(target_size[1] / aspect))
            else:
                new_size = (int(target_size[0] * aspect), target_size[1])
            
            img = img.resize(new_size, Image.LANCZOS)
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True, progressive=True)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Image optimization error: {str(e)}")
            raise Exception(f"Image optimization failed: {str(e)}")

    @staticmethod
    @lru_cache(maxsize=1000)
    def get_image_hash(image_bytes):
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('L').resize((8, 8), Image.LANCZOS)
        pixels = np.array(img.getdata(), dtype=np.float64).reshape((8, 8))
        dct = np.abs(np.fft.dct(np.fft.dct(pixels, axis=0), axis=1))
        return hash(dct.tobytes())

def get_drive_service():
    if not hasattr(thread_local, "drive_service"):
        try:
            credentials = Credentials.from_service_account_info(CREDENTIALS, scopes=SCOPES)
            thread_local.drive_service = build('drive', 'v3', credentials=credentials)
        except Exception as e:
            logger.error(f"Failed to initialize Drive service: {e}")
            raise
    return thread_local.drive_service

def get_rekognition_client():
    if not hasattr(thread_local, "rekognition_client"):
        try:
            thread_local.rekognition_client = boto3.client(
                'rekognition',
                aws_access_key_id=AWS_CREDENTIALS['aws_access_key_id'],
                aws_secret_access_key=AWS_CREDENTIALS['aws_secret_access_key'],
                region_name=AWS_CREDENTIALS['region']
            )
        except Exception as e:
            logger.error(f"Failed to initialize Rekognition client: {e}")
            raise
    return thread_local.rekognition_client

def list_drive_images():
    try:
        query = f"'{DRIVE_FOLDER_ID}' in parents and trashed = false and (mimeType contains 'image/')"
        results = get_drive_service().files().list(
            q=query,
            fields="files(id, name, mimeType, createdTime, size)",
            orderBy="createdTime desc",
            pageSize=100
        ).execute()
        return results.get('files', [])
    except Exception as e:
        logger.error(f"Error listing Drive images: {e}")
        return []

def process_single_comparison(source_bytes, drive_file, processor):
    try:
        fh = io.BytesIO()
        request = get_drive_service().files().get_media(fileId=drive_file['id'])
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        target_bytes = fh.getvalue()
        optimized_target = processor.optimize_image(target_bytes)
        
        comparison = get_rekognition_client().compare_faces(
            SourceImage={'Bytes': source_bytes},
            TargetImage={'Bytes': optimized_target},
            SimilarityThreshold=70
        )
        
        if comparison['FaceMatches']:
            return {
                'filename': drive_file['name'],
                'file_id': drive_file['id'],
                'similarity': comparison['FaceMatches'][0]['Similarity']
            }
            
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        return None
    finally:
        if 'fh' in locals():
            fh.close()
    
    return None

def stream_results(generator):
    """Helper function to stream JSON responses"""
    for item in generator:
        yield json.dumps(item) + '\n'

def process_batch(batch, optimized_source, processor, progress_queue):
    """Process a batch of images and return results"""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        future_to_file = {
            executor.submit(
                process_single_comparison,
                optimized_source,
                drive_file,
                processor
            ): drive_file for drive_file in batch
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            progress_queue.put(1)
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error in batch comparison: {e}")
                continue
                
    return results

def process_image_stream(image_bytes):
    """Generator function to process images in parallel batches"""
    processor = OptimizedImageProcessor()
    optimized_source = processor.optimize_image(image_bytes)
    progress_queue = Queue()
    
    try:
        source_response = get_rekognition_client().detect_faces(
            Image={'Bytes': optimized_source},
            Attributes=['DEFAULT']
        )

        if not source_response['FaceDetails']:
            yield {'type': 'error', 'message': 'No face detected in the uploaded image'}
            return

        drive_images = list_drive_images()
        total_images = len(drive_images)
        processed = 0
        
        all_batches = [
            drive_images[i:i + BATCH_SIZE]
            for i in range(0, len(drive_images), BATCH_SIZE)
        ]
        
        for i in range(0, len(all_batches), NUM_PARALLEL_BATCHES):
            current_batches = all_batches[i:i + NUM_PARALLEL_BATCHES]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PARALLEL_BATCHES) as executor:
                future_to_batch = {
                    executor.submit(
                        process_batch,
                        batch,
                        optimized_source,
                        processor,
                        progress_queue
                    ): batch for batch in current_batches
                }
                
                pending_futures = set(future_to_batch.keys())
                
                while pending_futures:
                    while not progress_queue.empty():
                        progress_queue.get()
                        processed += 1
                        if processed % 5 == 0:
                            yield {
                                'type': 'progress',
                                'processed': processed,
                                'total': total_images
                            }
                    
                    done_futures, pending_futures = concurrent.futures.wait(
                        pending_futures,
                        timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done_futures:
                        try:
                            batch_results = future.result()
                            for result in batch_results:
                                yield {
                                    'type': 'match',
                                    'data': result
                                }
                        except Exception as e:
                            logger.error(f"Error processing batch: {e}")
                            continue

    except Exception as e:
        logger.error(f"Error in face comparison: {e}")
        yield {'type': 'error', 'message': str(e)}

# API Routes
@app.route('/')
def root():
    """Root endpoint for health checks"""
    return jsonify({
        "status": "online",
        "service": "PicFinder API",
        "timestamp": time.time()
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time()
    })

@app.route('/set-folder-id', methods=['POST', 'OPTIONS'])
def set_folder_id():
    try:
        if request.method == 'OPTIONS':
            return '', 204  # Respond to preflight request
            
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Content-Type must be application/json'
            }), 415

        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
            
        folder_id = data.get('folderId')
        
        if not folder_id:
            return jsonify({
                'status': 'error',
                'message': 'No folder ID provided'
            }), 400

        # Validate folder ID exists in Drive
        try:
            drive_service = get_drive_service()
            drive_service.files().get(fileId=folder_id).execute()
        except Exception as e:
            logger.error(f"Invalid folder ID: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid folder ID'
            }), 400
            
        global DRIVE_FOLDER_ID
        DRIVE_FOLDER_ID = folder_id
        
        response = jsonify({
            'status': 'success',
            'message': 'Folder ID set successfully',
            'folderId': folder_id
        })
        
        return response, 200
        
    except Exception as e:
        logger.error(f"Error in set_folder_id: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/search-face', methods=['POST'])
def search_face():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        image_bytes = file.read()
        if len(image_bytes) > MAX_IMAGE_SIZE:
            return jsonify({'error': 'Image size must be less than 15MB'}), 400

        return Response(
            stream_results(process_image_stream(image_bytes)),
            mimetype='application/x-ndjson'
        )

    except Exception as e:
        logger.error(f"Error in search_face: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/image/<file_id>')
def serve_image(file_id):
    try:
        # Try to get from cache first
        cached_image = image_cache.get(f"img_{file_id}")
        if cached_image:
            return send_file(
                io.BytesIO(cached_image),
                mimetype='image/jpeg',
                cache_timeout=CACHE_TTL
            )

        request = get_drive_service().files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        fh.seek(0)
        image_bytes = fh.getvalue()

        try:
            processor = OptimizedImageProcessor()
            optimized_image = processor.optimize_image(
                image_bytes,
                target_size=(1200, 1200),
                quality=90
            )
            
            image_cache.set(f"img_{file_id}", optimized_image)
            
            return send_file(
                io.BytesIO(optimized_image),
                mimetype='image/jpeg',
                cache_timeout=CACHE_TTL
            )
        except Exception as e:
            logger.warning(f"Image optimization failed, serving original: {e}")
            return send_file(
                fh, 
                mimetype='image/jpeg',
                cache_timeout=CACHE_TTL
            )
            
    except Exception as e:
        logger.error(f"Error serving image {file_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to retrieve image: {str(e)}"
        }), 404
    finally:
        if 'fh' in locals():
            try:
                fh.close()
            except:
                pass

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Error handlers with enhanced logging and consistent response format
@app.errorhandler(400)
def bad_request_error(error):
    logger.warning(f"400 error: {request.url} - {error}")
    return jsonify({
        "status": "error",
        "error": "Bad Request",
        "message": str(error),
        "path": request.path
    }), 400

@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({
        "status": "error",
        "error": "Not Found",
        "message": "The requested resource was not found",
        "path": request.path
    }), 404

@app.errorhandler(405)
def method_not_allowed_error(error):
    logger.warning(f"405 error: {request.url} - {request.method}")
    return jsonify({
        "status": "error",
        "error": "Method Not Allowed",
        "message": f"The {request.method} method is not allowed for this endpoint",
        "path": request.path
    }), 405

@app.errorhandler(413)
def payload_too_large_error(error):
    logger.warning(f"413 error: {request.url}")
    return jsonify({
        "status": "error",
        "error": "Payload Too Large",
        "message": "The uploaded file exceeds the maximum allowed size",
        "path": request.path
    }), 413

@app.errorhandler(415)
def unsupported_media_type_error(error):
    logger.warning(f"415 error: {request.url}")
    return jsonify({
        "status": "error",
        "error": "Unsupported Media Type",
        "message": "The provided content type is not supported",
        "path": request.path
    }), 415

@app.errorhandler(429)
def too_many_requests_error(error):
    logger.warning(f"429 error: {request.url}")
    return jsonify({
        "status": "error",
        "error": "Too Many Requests",
        "message": "Rate limit exceeded. Please try again later",
        "path": request.path
    }), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {error}")
    return jsonify({
        "status": "error",
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "details": str(error) if app.debug else None,
        "path": request.path
    }), 500

@app.errorhandler(503)
def service_unavailable_error(error):
    logger.error(f"503 error: {error}")
    return jsonify({
        "status": "error",
        "error": "Service Unavailable",
        "message": "The service is temporarily unavailable",
        "path": request.path
    }), 503

# Initialize global variables
DRIVE_FOLDER_ID = None

def init_services():
    """Initialize all required services"""
    try:
        # Initialize Google Drive service
        credentials = Credentials.from_service_account_info(CREDENTIALS, scopes=SCOPES)
        drive_service = build('drive', 'v3', credentials=credentials)
        
        # Initialize AWS Rekognition client
        rekognition_client = boto3.client(
            'rekognition',
            aws_access_key_id=AWS_CREDENTIALS['aws_access_key_id'],
            aws_secret_access_key=AWS_CREDENTIALS['aws_secret_access_key'],
            region_name=AWS_CREDENTIALS['region']
        )
        
        logger.info("Services initialized successfully")
        return True
        
    except Exception as e:
        logger.critical(f"Error during service initialization: {e}")
        return False

if __name__ == '__main__':
    # Initialize services
    if not init_services():
        logger.critical("Failed to initialize services")
        raise SystemExit("Application startup failed")
    
    # Get configuration from environment variables
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Configure SSL if certificates are provided
    ssl_context = None
    ssl_cert = os.getenv('SSL_CERT')
    ssl_key = os.getenv('SSL_KEY')
    if ssl_cert and ssl_key:
        ssl_context = (ssl_cert, ssl_key)
        logger.info("SSL configured")
    
    # Start the application
    logger.info(f"Starting application on {host}:{port} (Debug: {debug})")
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True,
        ssl_context=ssl_context
    )
