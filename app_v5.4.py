def extract_text(response) -> str:
    """
    Extrahiert den Antworttext sicher, egal ob response.content 
    ein String oder eine Liste von Content-Blöcken ist.
    """
    content = response.content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif hasattr(part, "text"):
                text_parts.append(part.text)
        return "".join(text_parts).strip()
    return content.strip()
