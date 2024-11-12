from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath('../'))
import plots_to_pdf
import numpy as np
import sklearn.naive_bayes as nb
import sklearn.svm as svm
import sklearn.neural_network as nn
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingRandomSearchCV

def halving_random_search(filename='student_learning.pdf'):
    import seaborn as sns
    try:
        df = pd.read_csv(os.path.abspath('../student/Student_performance_data.csv'))
    except:
        df = pd.read_csv(os.path.abspath('student/Student_performance_data.csv'))
    np.random.seed(42)
    df = df.dropna()
    df= df.drop(columns='StudentID')
    categorical_columns = ['Gender','Ethnicity','ParentalSupport','ParentalEducation']
    for i in categorical_columns:
        df[i] = df[i].astype('object')
    df = pd.get_dummies(df,drop_first=True)
    X = df.drop('GradeClass', axis=1)
    y = df['GradeClass']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

    models ={
        'Naive Bayes': {
            'model': nb.GaussianNB(),
            'params':{
                'priors':[None, [0.01,0.09,0.1,0.3,0.5]],
                'var_smoothing':[1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
            }
        },
        'SVM': {
            'model': svm.SVC(random_state=42),
            'params':{
                'C':np.linspace(0.001,100,100),
                'kernel':['linear','poly','rbf','sigmoid']
            }
        },
        'Neural Network': {
            'model': nn.MLPClassifier(random_state=42),
            'params':{
                'hidden_layer_sizes':[(100,),(10,16,32,32,16,10),(8,16,32,20,10,8),(20,40,20),(32,50,50,32)],
                                        #    (12,24,64,24,8),(40,50,40),(50,60,50),(32,64,128,256,128,64,32)],
                'learning_rate_init':np.linspace(0.0001,1,100)
            }
        }
    }

    best_params = {}
    scores = {}
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import make_scorer
    cohen_kappa_scorer = make_scorer(cohen_kappa_score,weights='quadratic')
    for model,config in models.items():
        clf = config['model']
        params = config['params']
        halving_cv = HalvingRandomSearchCV(clf,params,cv=cv,scoring=cohen_kappa_scorer,n_jobs=-1,verbose=2,random_state=42)
        halving_cv.fit(X_train,y_train)
        best_params[model] = halving_cv.best_params_
        scores[model] = halving_cv.best_score_

    print(best_params)
    print(scores)

    with open('student_halving_RF_results.txt', 'w') as f:
        for name, params in best_params.items():
            f.write(f"\n{name} best hyperparameters:\n")
            for param, value in list(params.items()):
                f.write(f"{param}= {value}\n")
        f.write("\nBest scores for each model:\n")
        for name, score in scores.items():
            f.write(f"{name}: {score}\n")

     #plotting the learning curve for each model with the best parameters
    figure=[]
    mean_fit_times = {}
    mean_train_scores = {}
    for name, config in models.items():
        model = config['model'].set_params(**best_params[name])
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(model, X_train, y_train,scoring=cohen_kappa_scorer, 
                                                                cv=cv,shuffle=True, n_jobs=-1,random_state=42,
                                                                train_sizes=np.linspace(0.1, 1.0, 100),
                                                                return_times=True)
        fig,ax=plt.subplots(figsize=(8, 6))
        mean_fit_times[name] = np.mean(fit_times)
        mean_train_scores[name] = np.mean(train_scores)
        ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
        ax.legend(loc="best")
        ax.grid(True)
        ax.set_title(f'Learning Curve for {name}')
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Cohen Kappa Score')
        figure.append(fig)
                  
        
    for i,j in best_params.items():
            #round values
            for key,value in j.items():
                if j[key] == None:
                    j[key] = value
                elif key == 'kernel' or key == 'hidden_layer_sizes':
                    j[key] = value
                else:
                    j[key]=round(value,5)

    with open('student_train_test_set_scores.txt', 'w') as f:
        for name,config in models.items():
            test_model = config['model'].set_params(**best_params[name])
            temp = []
            for key,value in best_params[name].items():
                temp.append(f"{key}={value}")
            test_model.fit(X_train,y_train)
            y_pred = test_model.predict(X_test)
            c_k = cohen_kappa_score(y_test, y_pred,weights='quadratic')
            c_k = round(c_k,3)
            #plot confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            fig,ax=plt.subplots(figsize=(8, 6))
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{name} - Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            figure.append(fig)


            print(f"\n\n\nModel: {name},\nBest Parameters: {temp}, \nTime to Train: {mean_fit_times[name]},\n Training Cohen Kappa Score: {mean_train_scores[name]},\nCohen Kappa TEST Score: {c_k}")
            f.write(f"\n\n\nModel: {name},\nBest Parameters: {temp}, \nTime to Train: {mean_fit_times[name]},\n Training Cohen Kappa Score: {mean_train_scores[name]},\nCohen Kappa TEST Score: {c_k}")
            
        

    
    import plots_to_pdf
    plots_to_pdf.to_pdf(figure,filename)
if __name__ == '__main__':
    import datetime
    filename = datetime.datetime.now().strftime("student_RF_%Y-%m-%d_%H-%M-%S.pdf")
    halving_random_search(filename)
