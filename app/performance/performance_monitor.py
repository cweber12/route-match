# app/performance/performance_monitor.py
# -------------------------------------------------------------
# A decorator to monitor performance of functions, especially
# those involved in frame generation and processing.
# Logs execution time and estimates frames per second.
# -------------------------------------------------------------
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def performance_monitor(func):

    def wrapper(*args, **kwargs):
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        logger.info(f"Performance Stats for {func.__name__}:")
        logger.info(f"  Execution time: {execution_time:.2f} seconds")
        
        # Estimate frames per second if this was frame generation
        if 'frame' in func.__name__.lower() and execution_time > 0:
            # Try to estimate frame count from args/kwargs
            frame_count = 0
            for arg in args:
                if isinstance(arg, dict) and hasattr(arg, '__len__'):
                    frame_count = len(arg)
                    break
            
            if frame_count > 0:
                fps = frame_count / execution_time
                logger.info(f"  Frame generation rate: {fps:.1f} frames/second")
                logger.info(f"  Average time per frame: {execution_time/frame_count:.3f} seconds")
        
        return result
    return wrapper