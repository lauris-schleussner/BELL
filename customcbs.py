from tensorflow.keras.callbacks import Callback

class StopIfValAccTanked(Callback):

    def __init__(self, sleep_epochs=0, val_acc_th=.11):
        super().__init__()
        self.val_acc_th = val_acc_th
        self.sleep_epochs = sleep_epochs
        self.cb_epoch_counter = 0

    def on_epoch_end(self, batch, logs={}):
        
        if (logs.get('val_accuracy') <= self.val_acc_th) and (self.cb_epoch_counter >= self.sleep_epochs):
                self.model.stop_training = True
        
        self.cb_epoch_counter += 1