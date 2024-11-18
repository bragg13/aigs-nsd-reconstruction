def log(message, type='LOG'):
    if isinstance(message, str):
        print(f"[{type}] {message}")
    else:
        # if it's an object
        print(f"[{type}] {message}")
