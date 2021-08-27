# # In this file we will implement the training of the dataset
# # we need: class Trainer(obj)

from cryptotradingindicator.data import *
from cryptotradingindicator.utils import *
from cryptotradingindicator.model import get_model
from sklearn.pipeline import Pipeline
import joblib
import termcolor

from tensorflow.keras.models import save_model



# #WORKFLOW
# x_train, ytrain = get_xy(get_train_data)
# trainer = Trainer()
# trainer.fit(x_train,ytrain,......)
# x_gecko,y_gecko = get_xy(get_coingecko)
# trainer.predict



class Trainer(object):
    def __init__(self):
        """
            X: pandas DataFrame
            y: pandas Series
        """

    def train(self):
        data_train = feature_engineer(get_train_data())
        data_train_scaled, self.scaler = minmaxscaling(data_train[['log_close']])

        # Split the data into x_train and y_train data sets
        self.x_train, self.y_train = get_xy(data_train_scaled, 60, 1)

        # Train the model
        self.model = get_model(self.x_train)

        self.model.fit(self.x_train,
                  self.y_train,
                  batch_size=8,
                  epochs=1,
                  validation_split=0.4)

        save_model(self.model, '../model.joblib')
        print(termcolor.colored("saved the model locally", "green"))

    def predict(self):
        x_gecko = get_xgecko(60,1)
        predictions = self.model.predict(x_gecko)
        predictions = self.scaler.inverse_transform(predictions)
        return np.exp(predictions)


#        joblib.dump(self.model, '../model.joblib')
#        print(termcolor.colored("model.joblib saved locally", "green"))

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    print("finished training and saved")
    predictions = trainer.predict()
    print(predictions)
    print("lets go to the moon!")











    # self.pipeline = Pipeline([
    #     ('preproc', preproc_pipe),
    #     ('LSTM', model())
    # ])

    #     def run(self):
    #         self.set_pipeline()
    #         self.pipeline.fit(self.X, self.y)

    #     def evaluate(self, X_test, y_test):
    #         """evaluates the pipeline on df_test and return the AME"""
    #         y_pred = self.pipeline.predict(X_test)
    #-->         mae = compute_rmse(y_pred, y_test)
    #         return round(rmse, 2)

    # def save_model_locally(self):
    #     """Save the model into a .joblib format"""
    #     joblib.dump(self.pipeline, 'model.joblib')
    #     print(colored("model.joblib saved locally", "green"))

#--> predict
#--> unscale

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
