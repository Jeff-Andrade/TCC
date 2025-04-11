import pandas as pd
from config import POS_CSV, NEG_CSV


def load_data():
    """Loads and labels positive and negative data from CSV files."""
    pos_df = pd.read_csv(POS_CSV)
    pos_df = pos_df[pos_df['tfopwg_disp'].isin(['CP', 'KP'])]
    pos_df['label'] = 'planet'
    pos_df['fa_label'] = 'not_fa'

    neg_df = pd.read_csv(NEG_CSV)
    neg_df = neg_df[neg_df['tfopwg_disp'].isin(['FA', 'FP'])]
    neg_df['label'] = neg_df['tfopwg_disp'].map({'FA': 'na', 'FP': 'not_planet'})
    neg_df['fa_label'] = neg_df['tfopwg_disp'].map({'FA': 'fa', 'FP': 'not_fa'})

    combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
    return combined_df