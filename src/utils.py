from feature_extractor import extract_features

def extract_features_wrapper(row):
    """
    A wrapper for feature extraction that works with a DataFrame row.
    Returns a tuple: (feature_vector, label) or None if extraction fails.
    """
    if row['label'] == 'na':
        return None
    f = extract_features(row['tid'], row['label'])
    if f:
        return f, 1 if row['label'] == 'planet' else 0
    return None