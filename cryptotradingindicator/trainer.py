from cryptotradingindicator.data import *
from cryptotradingindicator.utils import *
from cryptotradingindicator.model import get_model
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import save_model
import joblib
import termcolor
from cryptotradingindicator.params import *


class Trainer(object):
    def __init__(self):
        """
        This trainer has two options: train and predict.
        """

    def train(self):
        data_train = feature_engineer(get_train_data())
        data_train_scaled, self.scaler = minmaxscaling(data_train[[CLOSE]])

        # Split the data into x_train and y_train data sets
        self.x_train, self.y_train = get_xy(data_train_scaled, LENGTH, HORIZON)

        # Train the model
        self.model = get_model(self.x_train)

        self.model.fit(self.x_train,
                  self.y_train,
                  batch_size=8,
                  epochs=10,
                  validation_split=0.4)

        #save_model(self.model, '../model.joblib')
        #print(termcolor.colored("saved the model locally", "green"))

        joblib.dump(self.model, '../model.joblib')
        print(termcolor.colored("model.joblib saved locally", "green"))

    def predict(self):
        x_gecko = get_xgecko(60,1)
        predictions = self.model.predict(x_gecko)
        predictions = self.scaler.inverse_transform(predictions)
        return np.exp(predictions)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    print("finished training and saved")
    predictions = trainer.predict()
    print(predictions)
    print("lets go to the moon!")

# if __name__ == "__main__":
#     # Get and clean data
#     df = get raw data
#     df = --> pipline clean_data(df)
#     y = --> define y (Friedas Notebook)
#     X = --> define X (Friedas Notebook)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#     # Train and save model, locally and
#     trainer = Trainer(X=X_train, y=y_train)
#     trainer.set_experiment_name('xp2')
#     trainer.run()
#     mae = trainer.evaluate(X_test, y_test)
#     print(f"mae: {mae}")
#     trainer.save_model_locally()
#-->     storage_upload()
