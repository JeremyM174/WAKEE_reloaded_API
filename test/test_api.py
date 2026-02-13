"""
Tests Pytest pour l'API Wakee
Vérifie les endpoints /health et /predict
"""

import pytest
import requests
from PIL import Image
import io
import time

API_URL="https://mevelios-wakee-reloaded-api.hf.space"

# ============================================================================
# TESTS /root
# ============================================================================

def test_root_endpoint_returns_200():
    """Vérifie que /root retourne 200 OK"""
    response = requests.get(f"{API_URL}", timeout=10)
    assert response.status_code == 200

# ============================================================================
# TESTS /test
# ============================================================================

def test_test_endpoint_returns_200():
    """Vérifie que /test retourne 200 OK"""
    response = requests.post(f"{API_URL}/test", timeout=10)
    assert response.status_code == 200

def test_test_endpoint_return_files():
    payload = b"Hello World !" 
    response = requests.post(f"{API_URL}/test", data=payload, headers={"Content-Type": "text/plain"}, timeout=10)
    assert response.content == payload


# ============================================================================
# CREATION image
# ============================================================================

# Crée une image test (224x224 RGB)
def create_test_image_bytes():
    img = Image.new('RGB', (224, 224), 'red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()

# ============================================================================
# TESTS /predict
# ============================================================================

def test_predict_endpoint_accepts_image():
    payload = create_test_image_bytes()
    response = requests.post(f"{API_URL}/predict", data=payload, headers={"Content-Type": "image/jpeg"}, timeout=30)
    assert response.status_code == 200

def test_predict_endpoint_return_str():
    payload = create_test_image_bytes()
    response = requests.post(f"{API_URL}/predict", data=payload, headers={"Content-Type": "image/jpeg"}, timeout=30)
    assert isinstance(response.text, str)

# ============================================================================
# TESTS /backup
# ============================================================================

def test_backup_endpoint_accepts_image():
    payload = create_test_image_bytes()
    response = requests.post(f"{API_URL}/backup", data=payload, headers={"Content-Type": "image/jpeg"}, timeout=30)
    assert response.status_code == 200

def test_backup_endpoint_return_str():
    payload = create_test_image_bytes()
    response = requests.post(f"{API_URL}/backup", data=payload, headers={"Content-Type": "image/jpeg"}, timeout=30)
    assert isinstance(response.text, str)