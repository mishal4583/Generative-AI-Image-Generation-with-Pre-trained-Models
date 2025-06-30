from flask import Flask, render_template, request, url_for, jsonify, send_file
from diffusers import StableDiffusionPipeline
import torch
import os
import json
from datetime import datetime
from collections import defaultdict
from flask_executor import Executor
import glob
import time
from threading import Thread, Lock
from flask import current_app # Import current_app for app context in threads
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# --- REGISTER THE DATETIME FORMAT FILTER HERE ---
def datetimeformat(value, format_str='%Y-%m-%d %H:%M:%S'):
    if isinstance(value, str): # Handle ISO string coming from saved history
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value # Return original if not a valid ISO format
    return value.strftime(format_str) if isinstance(value, datetime) else ""

app.add_template_filter(datetimeformat) # This is the correct way to add a filter
# ------------------------------------------------

app.config['UPLOAD_FOLDER'] = os.path.join("static", "images")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['EXECUTOR_TYPE'] = 'thread'
app.config['EXECUTOR_MAX_WORKERS'] = 4

# Initialize executor
executor = Executor(app)

# Ensure image folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Persistence Functions ---
HISTORY_FILE = 'history.json'
STATS_FILE = 'stats.json'

# Use a lock to protect concurrent access to history_data and usage_stats_data
# This is crucial for thread safety when multiple operations modify these globals
data_lock = Lock()

def load_data():
    """Loads history and usage stats from files, thread-safe."""
    with data_lock:
        app.logger.debug("Acquired lock for loading data.")
        current_history = []
        current_stats = defaultdict(int)

        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    loaded_history = json.load(f)
                    for item in loaded_history:
                        if 'timestamp' in item and isinstance(item['timestamp'], str):
                            try:
                                item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                            except ValueError:
                                app.logger.warning(f"Could not parse timestamp '{item['timestamp']}' for item: {item.get('prompt', 'N/A')}")
                                # Keep as string if parsing fails, so it doesn't break later.
                                # Alternatively, you could skip this item or set a default.
                        current_history.append(item)
                app.logger.info(f"Loaded {len(current_history)} items from {HISTORY_FILE}.")
            except json.JSONDecodeError:
                app.logger.warning(f"Could not decode {HISTORY_FILE}, starting fresh history.")
            except Exception as e:
                app.logger.error(f"Error loading history data: {e}")
        
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, 'r') as f:
                    loaded_stats = json.load(f)
                    current_stats.update(loaded_stats)
                app.logger.info(f"Loaded stats from {STATS_FILE}: {current_stats}")
            except json.JSONDecodeError:
                app.logger.warning(f"Could not decode {STATS_FILE}, starting fresh stats.")
            except Exception as e:
                app.logger.error(f"Error loading stats data: {e}")
        
        app.logger.debug("Released lock after loading data.")
        return current_history, current_stats

def save_data(history_to_save, stats_to_save):
    """Saves history and usage stats to files, thread-safe."""
    with data_lock:
        app.logger.debug("Acquired lock for saving data.")
        # Convert datetime objects to ISO format strings before saving history
        serializable_history = []
        for item in history_to_save:
            temp_item = item.copy()
            if 'timestamp' in temp_item and isinstance(temp_item['timestamp'], datetime):
                temp_item['timestamp'] = temp_item['timestamp'].isoformat()
            serializable_history.append(temp_item)

        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(serializable_history, f, indent=4)
            app.logger.info(f"Saved {len(serializable_history)} items to {HISTORY_FILE}.")
        except Exception as e:
            app.logger.error(f"Error saving history data: {e}")

        try:
            with open(STATS_FILE, 'w') as f:
                json.dump(dict(stats_to_save), f, indent=4) # Convert defaultdict to dict for saving
            app.logger.info(f"Saved stats to {STATS_FILE}.")
        except Exception as e:
            app.logger.error(f"Error saving stats data: {e}")
        app.logger.debug("Released lock after saving data.")


# --- Model Configuration ---
model_map = {
    "Realisian": {
        "id": "runwayml/stable-diffusion-v1-5",
        "max_size": 768,
        "dtype": torch.float16
    },
    "Waifu": {
        "id": "hakurei/waifu-diffusion",
        "max_size": 1024,
        "dtype": torch.float16
    },
    "OrangeMix": {
        # IMPORTANT: Replace with a proper OrangeMix setup:
        # Option 1 (Recommended if available as Diffusers repo):
        # "id": "WarriorMama777/OrangeMixs", "subfolder": "AbyssOrangeMix2/Diffusers",
        # Option 2 (If you downloaded a .safetensors file locally):
        # "id": "./models/OrangeMix/AbyssOrangeMix2_sfw.safetensors", "from_single_file": True,
        # For demonstration, falling back to Waifu Diffusion to ensure app runs:
        "id": "hakurei/waifu-diffusion", 
        "max_size": 1024,
        "dtype": torch.float16
    }
}

# Cache loaded pipelines
loaded_pipelines = {}

# Global variable to hold the cleanup thread instance
cleanup_thread_instance = None

# --- Background Cleanup Thread ---
def cleanup_old_files_task():
    """Background task to clean up old files"""
    with app.app_context(): # Essential for logging in background thread
        while True:
            try:
                files = glob.glob(os.path.join(current_app.config['UPLOAD_FOLDER'], '*.png'))
                for f in files:
                    # Keep files for 7 days (604800 seconds), adjust as needed
                    if os.path.getmtime(f) < time.time() - 604800:
                        os.remove(f)
                        current_app.logger.info(f"Cleaned up old file: {f}")
            except Exception as e:
                current_app.logger.error(f"Cleanup error: {str(e)}")
            time.sleep(3600)  # Run hourly

# Function to start the cleanup thread, ensuring it only starts once
def start_cleanup_thread():
    global cleanup_thread_instance
    if cleanup_thread_instance is None or not cleanup_thread_instance.is_alive():
        cleanup_thread_instance = Thread(target=cleanup_old_files_task, daemon=True, name='cleanup_thread')
        cleanup_thread_instance.start()
        app.logger.info("Cleanup thread started.")
    else:
        app.logger.info("Cleanup thread is already running (via Flask reloader).")

# --- Core Functions ---
def get_pipeline(model_name):
    """Get cached pipeline or load new one"""
    if model_name not in loaded_pipelines:
        model_config = model_map.get(model_name)
        if not model_config:
            raise ValueError(f"Invalid model selected: {model_name}")
            
        try:
            # Check if loading from a single file (like a downloaded .safetensors/.ckpt)
            if model_config.get("from_single_file"):
                pipe = StableDiffusionPipeline.from_single_file(
                    model_config["id"],
                    torch_dtype=model_config["dtype"],
                    # If it's a .safetensors file, use use_safetensors=True
                    use_safetensors=model_config["id"].endswith(".safetensors")
                )
            else:
                # Load from Hugging Face model ID (with optional subfolder)
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_config["id"],
                    torch_dtype=model_config["dtype"],
                    use_safetensors=True, # Prefer safetensors if available in the HF repo
                    subfolder=model_config.get("subfolder") # Pass subfolder if specified
                )
            pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            loaded_pipelines[model_name] = pipe
            app.logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            app.logger.error(f"Model loading error for {model_name}: {e}")
            raise
    return loaded_pipelines[model_name]

def validate_input(prompt, model, width, height):
    """Validate user input"""
    if not prompt or len(prompt) > 1000:
        raise ValueError("Prompt must be between 1-1000 characters.")
    
    if model not in model_map:
        raise ValueError("Invalid model selected.")
    
    max_size = model_map[model]["max_size"]
    # Check if width/height are integers first to prevent ValueError
    if not (isinstance(width, int) and isinstance(height, int)):
        raise ValueError("Width and Height must be integers.")
    if not (256 <= width <= max_size and 256 <= height <= max_size):
        raise ValueError(f"Dimensions must be between 256x256 and {max_size}x{max_size} for {model}.")
    
# --- Async Image Generation Task ---
def generate_image_async(prompt, model, width, height, filename, filepath):
    """Background task for image generation. Needs app_context for url_for."""
    with app.app_context(): # Acquire app context for url_for and logger
        try:
            app.logger.info(f"Starting image generation for prompt: '{prompt[:50]}...' using model '{model}'")
            pipe = get_pipeline(model)
            image = pipe(prompt, height=height, width=width).images[0]
            image.save(filepath)
            
            static_path = os.path.join("images", filename).replace("\\", "/")
            static_url = url_for('static', filename=static_path)
            
            # Load current state, update, and save, all within the lock
            current_history, current_stats = load_data() 
            
            new_entry = {
                "model": model,
                "prompt": prompt,
                "image_url": static_url,
                "timestamp": datetime.now()
            }
            current_history.insert(0, new_entry) # Add to the beginning
            current_history = current_history[:10] # Keep last 10 entries

            # Update stats (this should have been updated in the main thread already,
            # but we load/save to ensure consistency across threads)
            # This update is mostly redundant if the main thread already updated it,
            # but it ensures the file is consistent with the latest image generation.
            # For simplicity, we just re-save what was committed by the main thread.
            # A more robust system might use a queue or separate mechanisms for stats.
            
            save_data(current_history, current_stats) 
            app.logger.info(f"Successfully generated and saved: {filepath}. History and stats updated.")

        except Exception as e:
            app.logger.error(f"Async image generation failed for prompt '{prompt[:50]}...': {e}", exc_info=True) # exc_info for full traceback

# --- Flask Routes ---
@app.route("/", methods=["GET"])
def index():
    history, stats = load_data() # Load current data
    return render_template(
        "index.html",
        prompt="",
        history=history,
        stats=dict(stats), # Ensure it's a dict for template
        error=None
    )

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.form["prompt"].strip()
    model = request.form["model"]
    aspect_ratio_val = request.form.get("aspect_ratio", "512x512")
    
    width, height = 512, 512 # Default
    try:
        if aspect_ratio_val == "custom":
            width = int(request.form.get("width", 512))
            height = int(request.form.get("height", 512))
        else:
            width, height = map(int, aspect_ratio_val.split("x"))
    except ValueError:
        app.logger.error("Invalid aspect ratio or custom dimensions provided.")
        return jsonify({"success": False, "error": "Invalid aspect ratio or custom dimensions."}), 400

    try:
        validate_input(prompt, model, width, height)
        
        # Load current data, update stats, and save
        current_history, current_stats = load_data()
        current_stats[model] += 1
        current_stats['total'] += 1
        save_data(current_history, current_stats) # Save updated stats immediately

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{model}_{timestamp_str}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        executor.submit(
            generate_image_async,
            prompt, model, width, height, filename, filepath
        )
        
        app.logger.info(f"Generation request submitted for '{prompt[:50]}...' with model '{model}'.")
        return jsonify({"success": True, "message": "Image generation started. Please check history shortly."}), 202
        
    except ValueError as ve: # Catch specific validation errors
        app.logger.warning(f"Validation error in /generate: {ve}")
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e: # Catch other unexpected errors
        app.logger.error(f"Unexpected error in /generate route: {e}", exc_info=True)
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500

@app.route("/delete-history", methods=["POST"])
def delete_history():
    if "image_url" not in request.form:
        app.logger.error("image_url missing in delete-history request.")
        return jsonify({"success": False, "error": "image_url is missing from request."}), 400

    image_url_to_delete = request.form["image_url"]
    
    current_history, current_stats = load_data() # Load current state

    original_history_len = len(current_history)
    updated_history = [h for h in current_history if h.get("image_url") != image_url_to_delete]
    deleted_from_list = original_history_len > len(updated_history)

    if not deleted_from_list:
        app.logger.warning(f"Attempted to delete image_url '{image_url_to_delete}' but it was not found in history_data.")

    # Reconstruct the absolute path from the URL
    file_deleted = False
    image_disk_path = None
    try:
        relative_image_path = image_url_to_delete.replace(url_for('static', filename=''), '', 1)
        image_disk_path = os.path.join(app.root_path, 'static', relative_image_path)

        if os.path.exists(image_disk_path):
            os.remove(image_disk_path)
            app.logger.info(f"Deleted file: {image_disk_path}")
            file_deleted = True
        else:
            app.logger.warning(f"File not found for deletion: {image_disk_path}")
    except Exception as e:
        app.logger.error(f"Error deleting physical file {image_disk_path}: {e}", exc_info=True)
        # Even if file deletion fails, we should still try to update history/stats
        # This error is returned to the client later.

    save_data(updated_history, current_stats) # Save updated history (stats are unchanged here)

    if file_deleted or deleted_from_list:
        return jsonify({"success": True, "message": "History item and/or file deleted successfully."})
    else:
        return jsonify({"success": False, "error": "Failed to find or delete item."}), 404


@app.route("/get-history", methods=["GET"])
def get_history():
    """Endpoint to fetch current history data."""
    history, _ = load_data() # Load current history
    # Convert datetime objects to ISO format strings for JSON serialization
    serializable_history = []
    for item in history:
        temp_item = item.copy()
        if 'timestamp' in temp_item and isinstance(temp_item['timestamp'], datetime):
            temp_item['timestamp'] = temp_item['timestamp'].isoformat()
        serializable_history.append(temp_item)
    app.logger.info(f"Returning {len(serializable_history)} history items.")
    return jsonify(serializable_history)

@app.route("/get-stats", methods=["GET"])
def get_stats():
    """Endpoint to fetch current usage stats."""
    _, stats = load_data() # Load current stats
    app.logger.info(f"Returning stats: {dict(stats)}")
    return jsonify(dict(stats)) # Ensure it's a regular dict for jsonify

if __name__ == "__main__":
    # Initial load of data when the app starts
    # Global variables history_data and usage_stats_data are NOT used directly anymore,
    # as load_data/save_data now manage the file interaction and return/take copies.
    # This prevents accidental direct modification of stale global variables.
    
    # Start cleanup thread only once
    start_cleanup_thread()
    app.run(debug=True, threaded=True, use_reloader=False) 
    # use_reloader=False recommended when using background threads managed manually,
    # as reloader can start the thread twice. If you rely on reloader for dev,
    # ensure your start_cleanup_thread is robust to multiple calls.