from flask import Flask, request, jsonify, send_file
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
import gc
import multiprocessing

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with all necessary headers
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "max_age": 3600,
        "supports_credentials": True
    }
})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    gc.collect()  # Force garbage collection after each request
    return response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.readonly']
MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15MB
REKOGNITION_MAX_SIZE = 5 * 1024 * 1024  # 5MB AWS limit
BATCH_SIZE = 5  # Reduced batch size
CACHE_TTL = 3600
MAX_WORKERS = 2  # Reduced from 4 to 2

# Optimized Image Cache
class ImageCache:
    def __init__(self, max_size=50):
        self.cache = {}
        self.lock = threading.Lock()
        self.max_size = max_size

    def get(self, key):
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < CACHE_TTL:
                    return data
                else:
                    del self.cache[key]
                    gc.collect()
            return None

    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            self.cache[key] = (value, time.time())

image_cache = ImageCache()

# Thread-local storage
thread_local = threading.local()

# Optimized Image Processor
class OptimizedImageProcessor:
    def __init__(self, max_workers=MAX_WORKERS):
        self.max_workers = max_workers
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
    @staticmethod
    def optimize_image(img_bytes, target_size=(800, 800), quality=85):
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background

                aspect = img.size[0] / img.size[1]
                new_size = (
                    min(target_size[0], img.size[0]),
                    min(target_size[1], img.size[1])
                )
                
                if img.size != new_size:
                    img = img.resize(new_size, Image.LANCZOS)
                
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality, optimize=True, progressive=True)
                return output.getvalue()
        except Exception as e:
            raise Exception(f"Image optimization failed: {str(e)}")

    def __del__(self):
        self.thread_pool.shutdown(wait=False)

def get_drive_service():
    if not hasattr(thread_local, "drive_service"):
        credentials = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
        thread_local.drive_service = build('drive', 'v3', credentials=credentials)
    return thread_local.drive_service

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
        
        comparison = rekognition_client.compare_faces(
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

def compare_faces_with_collection(image_bytes):
    processor = OptimizedImageProcessor()
    optimized_source = processor.optimize_image(image_bytes)
    
    try:
        source_response = rekognition_client.detect_faces(
            Image={'Bytes': optimized_source},
            Attributes=['DEFAULT']
        )

        if not source_response['FaceDetails']:
            return {'error': 'No face detected in the uploaded image'}

        drive_images = list_drive_images()
        matches = []
        
        for i in range(0, len(drive_images), BATCH_SIZE):
            batch = drive_images[i:i + BATCH_SIZE]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_file = {
                    executor.submit(process_single_comparison, optimized_source, drive_file, processor): drive_file 
                    for drive_file in batch
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    try:
                        result = future.result(timeout=30)  # 30 second timeout per comparison
                        if result:
                            matches.append(result)
                    except Exception as e:
                        logger.error(f"Error in comparison: {e}")
                        continue

            time.sleep(0.1)  # Small delay between batches
            gc.collect()  # Garbage collection between batches

        return sorted(matches, key=lambda x: x['similarity'], reverse=True)

    except Exception as e:
        logger.error(f"Error in face comparison: {e}")
        return {'error': str(e)}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API Routes
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

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(compare_faces_with_collection, image_bytes)
            try:
                matches = future.result(timeout=90)
            except concurrent.futures.TimeoutError:
                return jsonify({'error': 'Operation timed out'}), 408

        if isinstance(matches, dict) and 'error' in matches:
            return jsonify({'error': matches['error']}), 400

        return jsonify({
            'matches': matches,
            'total_processed': len(list_drive_images())
        })

    except Exception as e:
        logger.error(f"Error in search_face: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        gc.collect()

@app.route('/set-folder-id', methods=['POST'])
def set_folder_id():
    global DRIVE_FOLDER_ID
    try:
        data = request.json
        DRIVE_FOLDER_ID = data.get('folderId')
        folder_metadata = get_drive_service().files().get(
            fileId=DRIVE_FOLDER_ID, 
            fields="name"
        ).execute()
        return jsonify({
            'success': True, 
            'folderName': folder_metadata.get('name')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/image/<file_id>')
def serve_image(file_id):
    try:
        cached_image = image_cache.get(f"img_{file_id}")
        if cached_image:
            return send_file(io.BytesIO(cached_image), mimetype='image/jpeg')

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
            return send_file(io.BytesIO(optimized_image), mimetype='image/jpeg')
        except Exception as e:
            logger.warning(f"Image optimization failed: {e}")
            return send_file(fh, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error serving image {file_id}: {e}")
        return jsonify({'error': str(e)}), 404
    finally:
        if 'fh' in locals():
            try:
                fh.close()
            except:
                pass
        gc.collect()

# Initialize services
try:
    credentials = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)
    
    aws_creds = json.load(open('credentials.json'))['aws']
    rekognition_client = boto3.client('rekognition',
        aws_access_key_id=aws_creds['aws_access_key_id'],
        aws_secret_access_key=aws_creds['aws_secret_access_key'],
        region_name=aws_creds['region']
    )
    
    DRIVE_FOLDER_ID = None
    
except Exception as e:
    logger.error(f"Error during initialization: {e}")
    raise

if __name__ == '__main__':
    # For development only - production will use gunicorn
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
