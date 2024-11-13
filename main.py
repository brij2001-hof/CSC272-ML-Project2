import datetime
import os

output_folder = ("./output/")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
derm_HyperTuning_filename = output_folder + datetime.datetime.now().strftime("derm_HyperTuning_%Y-%m-%d_%H-%M-%S.pdf")
student_HyperTuning_filename = output_folder + datetime.datetime.now().strftime("student_HyperTuning_%Y-%m-%d_%H-%M-%S.pdf")

'''
Evaluating hyperparameters for each model using validation curve.
Outputs (for each dataset):
        - .pdf | Plots containing validation curves for each hyper parameter tested for each model. Also confusion
'''
import derm.derm
import student.student
derm.derm.evaluate_parameters(filename=derm_HyperTuning_filename)
student.student.evaluate_parameters(filename=student_HyperTuning_filename)



''''''''''''''''''''''''''''''''''''
derm_learning_curves_filename = output_folder + datetime.datetime.now().strftime("derm_learning_curves_%Y-%m-%d_%H-%M-%S.pdf")
student_learning_curves_filename = output_folder + datetime.datetime.now().strftime("student_learning_curves_%Y-%m-%d_%H-%M-%S.pdf")
'''
Calculates best params using halving random search.
Fits on train set, predicts on test set and,
Outputs (for each dataset):
        - .pdf | Plots containing learning curves, confusion matrix and precision-recall AOC(*only for heart dataset)
        - .txt | Parameters returned by grid search for each model and their cross-validation scores
        - .txt | Test set scores, params and time to train for each model


'''
import derm.derm_RF
import student.student_RF
derm.derm_RF.halving_random_search(filename=derm_learning_curves_filename)
student.student_RF.halving_random_search(filename=student_learning_curves_filename)
