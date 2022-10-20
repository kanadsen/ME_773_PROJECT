'''
Federated Learning Works. Accuracy increased to 95%
Dataset needs to be divided and supplied as a separate python file which has not been done here. (Need separate implementation)
However accuracy has increased with federated learning.
'''
from packages_imp import*
import flwr as fl
import tensorflow as tf
# Import packages required for CNN_LSTM model
from keras.models import Sequential
from keras.layers import Dense, Conv1D   #Conv1D is used for 1D array shapes especially when data is stored in form of batches
from keras.layers import Dropout, Input
from keras import initializers
from keras.layers import MaxPooling1D
from keras.models import Model 
from keras.layers import TimeDistributed, Flatten
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from ipynb.fs.full.CNN_LSTM_fed import*

# defining accuracy
def Accuracy(true_RUL,preds_for_last_example):
  c=0
  N=15
  for i in range(len(true_RUL)):
      if(abs(true_RUL[i]-preds_for_last_example[i])<=N):
          c+=1
  acc=c/len(true_RUL)*100
  return str(acc)

new_model = tf.keras.models.load_model('Mymodel.h5')


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return new_model.get_weights()
    def fit(self, parameters, config):
        def scheduler(epoch):
          if epoch < 10:
             return 0.001
          else:
             return 0.0001
        callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1) 
        history = new_model.fit(processed_train_data, processed_train_targets, epochs = 10,
                    batch_size = 60)
        return new_model.get_weights(), len(processed_train_data), {}

    def evaluate(self, parameters, config):
        rul_pred = new_model.predict(processed_test_data).reshape(-1)
        preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])
        indices_of_last_examples = np.cumsum(num_test_windows_list) - 1
        preds_for_last_example = np.concatenate(preds_for_each_engine)[indices_of_last_examples]
        s=Accuracy(true_RUL,preds_for_last_example)
        print(s)  #Prints Accuracy
        loss=10.0
        l=10
        return float(s), l, {"accuracy": float(s)}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())