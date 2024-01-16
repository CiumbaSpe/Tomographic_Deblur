import copy

class EarlyStopping: 
    def __init__(self, patience = 5, min_delta = 0, restore_best_weight = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weight = restore_best_weight
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if(self.best_loss is None):
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif (self.best_loss - val_loss >= self.min_delta):            
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"improvement found, counter reset to  {self.counter}"
        else: 
            self.counter += 1 
            self.status = f"No improvement found in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs"
                if self.restore_best_weight:
                    model.load_state_dict(self.best_model)
                return True
        print(self.counter)
        return False