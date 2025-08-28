from hashlib import md5

def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()

def compute_content_hash(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()
