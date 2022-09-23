# plot tensorlfow model

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model








quit()



modelpath = "models/saved_cnn"
model = load_model(modelpath)
modelname = "cnn"

plot_model(model, to_file = "plots/" + modelname + ".png", show_layer_names = False, show_shapes = True)