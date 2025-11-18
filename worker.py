# worker.py (Run the worker with this file)
#!/usr/bin/env python
"""
Worker startup script - MUST import from app.py to ensure Flask context
Run with: python worker.py
"""
from app import celery

if __name__ == '__main__':
    celery.worker_main([
        'worker',
        '--loglevel=info',
        '--pool=threads',
        '--concurrency=1'
    ])
