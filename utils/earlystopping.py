import logging
import numpy as np

class EarlyStopping:
    def __init__(self, 
                 patience=3, 
                 delta=0.0, 
                 mode='min', 
                 verbose=True,
                 logger=None):
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = None
        self.mode = mode
        self.delta = delta

        self.logger = logger if logger is not None else logging.getLogger(__name__)
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    self.logger.info(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    self.logger.info(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                                     f'Best: {self.best_score:.5f}' \
                                     f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    self.logger.info(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    self.logger.info(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                                     f'Best: {self.best_score:.5f}' \
                                     f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
            
        if self.counter >= self.patience:
            if self.verbose:
                self.logger.info(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False