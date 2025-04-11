from feature_extractor import extract_features

def extract_features_wrapper(row):

    if row['label'] == 'na':
        return None
    f = extract_features(row['tid'], row['label'])
    if f:
        return f, 1 if row['label'] == 'planet' else 0
    return None