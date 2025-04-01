import pandas as pd


def merge_by_epochs(df: pd.DataFrame):
    merged = df.groupby("epoch").agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()

    # Save the cleaned version
    return merged


def get_best_epochs(df: pd.DataFrame):    
    best_dice_epoch = df['val_dice'].idxmax()
    best_loss_epoch = df['val_loss'].idxmin()
    
    best_dice_row = df.iloc[best_dice_epoch].copy()
    
    best_loss_row = df.iloc[best_loss_epoch].copy()
    
    return {
        'best_dice': best_dice_row,
        'best_loss': best_loss_row
    }
