
import time
import logging
import signal
import sys
import os

# Add parent directory to path if running from examples
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from aero.interface import AeroSystem
except ImportError:
    try:
        from interface import AeroSystem
    except ImportError:
        # Fallback if running from inside examples without package context
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from interface import AeroSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    logger.info("Interrupt received, stopping...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting Aero System Example")
    
    # Initialize system
    system = AeroSystem(config={"enable_visualization": True})
    
    try:
        system.initialize()
        system.start()
        
        logger.info("System running. Press Ctrl+C to stop.")
        
        while True:
            # Process frames as they arrive
            pose = system.process_next_frame(timeout=1.0)
            
            if pose:
                # Do something with the pose
                # logger.info(f"Pose: {pose.translation}")
                pass
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        system.stop()
        logger.info("System shutdown complete.")

if __name__ == "__main__":
    main()
