from hashlib import md5


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def compute_content_hash(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def compute_mm_hash(item, prefix: str = ""):
    if item.get("type") == "text" and item.get("text"):
        content = item["text"].strip()
    elif item.get("type") == "image" and item.get("img_path"):
        content = f"image:{item['img_path']}"
    elif item.get("type") == "table" and item.get("table_body"):
        content = f"table:{item['table_body']}"
    elif item.get("type") == "equation" and item.get("text"):
        content = f"equation:{item['text']}"
    else:
        content = str(item)
    return prefix + md5(content.encode()).hexdigest()


def compute_dict_hash(d: dict, prefix: str = ""):
    items = tuple(sorted(d.items()))
    return prefix + md5(str(items).encode()).hexdigest()
