import os
import uuid
import shutil

def setup_workspace(folder):
    request_id = str(uuid.uuid4())
    os.makedirs(folder, exist_ok=True)

    working_dir = os.path.join(folder, request_id)
    os.makedirs(working_dir, exist_ok=True)

    log_dir = os.path.join(folder, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{request_id}.log")

    return log_file, working_dir


def cleanup_workspace(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
