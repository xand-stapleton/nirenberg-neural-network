import os
import re
import ast

for fname in os.listdir('.'):
    if not fname.endswith(('.txt', '.yaml', '.yml', '.json')):  # adjust extensions
        continue

    with open(fname, 'r') as f:
        text = f.read()

    # ---- Replace "one positive domain" → "many positive domains"
    text = text.replace("one positive domain", "many positive domains")

    # ---- Replace cs: [...] → computed l(l+1)
    def repl(match):
        ls_str = match.group(1)
        # Parse the list literal safely
        ls = ast.literal_eval(ls_str)
        new_cs = [l * (l + 1) for l in ls]
        return f"cs: {new_cs}"

    # Look up the ls list and replace corresponding cs line
    # This finds: ls: [1, 2, 3] (with any spacing)
    ls_pattern = r"ls:\s*\[([^\]]+)\]"
    ls_match = re.search(ls_pattern, text)

    if ls_match:
        ls_list_str = "[" + ls_match.group(1) + "]"
        ls_list = ast.literal_eval(ls_list_str)
        new_cs = [l * (l + 1) for l in ls_list]

        # Now replace the cs line entirely
        text = re.sub(r"cs:\s*\[[^\]]*\]", f"cs: {new_cs}", text)

    # ---- Write back the file
    with open(fname, 'w') as f:
        f.write(text)

    print(f"Processed {fname}")

