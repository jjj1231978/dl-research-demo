import torch 

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve 
    after a given patience.
    """
    def __init__(self, savepath, patience=5, min_delta=0.0, verbose=False):
        """
        Args:
            patience  (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change in monitored value to be considered improvement.
            verbose   (bool): If True, prints a message for each update.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.savepath = savepath
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, model, val_loss):
        # If no best_loss set yet, treat current as best
        if self.best_loss is None:
            self.best_loss = val_loss
        # Check if there's an improvement
        elif (self.best_loss - val_loss) > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.savepath)
            if self.verbose:
                print(f"Validation loss improved. Resetting counter.")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True