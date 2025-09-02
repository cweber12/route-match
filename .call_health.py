from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.abspath('.'))
import importlib
m = importlib.import_module('app.main')
app = m.app
print('routes:', [r.path for r in app.routes])
print('route methods for /api/health:', [ (r.path, getattr(r, 'methods', None)) for r in app.routes if r.path=='/api/health'])
with TestClient(app) as client:
    res = client.get('/api/health')
    print('status', res.status_code, 'body', res.text)
