from typing import Dict


def get_category(doc) -> str:
    """
    Returns the document category of the document
    e.g. UKGPA, UKSI, etc
    """
    if 'primary' in doc['metadata']:
        category = doc['metadata']['primary']['document-main-type']
    else:
        category = doc['metadata']['secondary']['document-main-type']
    return category