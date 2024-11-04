import datetime


derm_HyperTuning_filename = datetime.datetime.now().strftime("derm_HyperTuning_%Y-%m-%d_%H-%M-%S.pdf")
zoo_HyperTuning_filename = datetime.datetime.now().strftime("zoo_HyperTuning_%Y-%m-%d_%H-%M-%S.pdf")

'''
Evaluating hyperparameters for each model using validation curve.
Outputs (for each dataset):
        - .pdf | Plots containing validation curves for each hyper parameter tested for each model. Also confusion
'''
import derm.derm
# import zoo.zoo
derm.derm.evaluate_parameters(filename=derm_HyperTuning_filename)
#zoo.zoo.evaluate_parameters(filename=zoo_HyperTuning_filename)



derm_learning_curves_filename = datetime.datetime.now().strftime("derm_learning_curves_%Y-%m-%d_%H-%M-%S.pdf")
zoo_learning_curves_filename = datetime.datetime.now().strftime("zoo_learning_curves_%Y-%m-%d_%H-%M-%S.pdf")



'''
Calculates best params using grid search.
Fits on train set, predicts on test set and,
Outputs (for each dataset):
        - .pdf | Plots containing learning curves, confusion matrix and precision-recall AOC(*only for heart dataset)
        - .txt | Parameters returned by grid search for each model and their cross-validation scores
        - .txt | Test set scores, params and time to train for each model


'''
# import GRID_churn
# import grid_heart
# grid_heart.gridsearch_learning_curves(filename=derm_learning_curves_filename) 
# GRID_churn.gridsearch_learning_curves(filename=zoo_learning_curves_filename)
