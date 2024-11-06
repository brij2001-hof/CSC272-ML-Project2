import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.naive_bayes as nb
import sklearn.svm as svm
import sklearn.neural_network as nn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, validation_curve
from sklearn.pipeline import Pipeline
import sys
import os
sys.path.append(os.path.abspath('../'))
import plots_to_pdf
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.model_selection import learning_curve

def evaluate_parameters(filename):
    import seaborn as sns
    df = pd.read_csv(os.path.abspath('../student/Student_performance_data.csv'))
    np.random.seed(42)
    df = df.dropna()
    df= df.drop(columns='StudentID')
     #correlation graph
    # matplotlib.use('TkAgg')
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(df.corr(),annot=True, cmap='coolwarm', fmt='.2f')
    # plt.tight_layout()
    # plt.savefig('correlation_matrix.png')
    # plt.close()
    # exit()
    
    X = df.drop('GradeClass', axis=1)  
    y = df['GradeClass']  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
    
    models ={
        # 'Naive Bayes': {
        #     'pipeline': Pipeline([('clf',nb.GaussianNB())]),
        #     'params':{
        #         # 'clf__priors':[None, [0.5,0.5], [0.7,0.3], [0.3,0.7]],
        #         'clf__var_smoothing':[1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
        #     }
        # },
        # 'SVM': {
        #     'pipeline': Pipeline([('clf',svm.SVC())]),
        #     'params':{
        #         'clf__C':[0.001,0.01,0.1,1,10,100],
        #         'clf__kernel':['linear','poly','rbf','sigmoid']
        #     }
        # },
        'Neural Network': {
            'pipeline': Pipeline([('clf',nn.MLPClassifier(random_state=42))]),
            'params':{
                'clf__hidden_layer_sizes':[(7,14,32,64,32,14,4),(20,30,20),(8,16,32,20,10,8),(12,24,64,24,8),(40,50,40),(50,60,50),(32,64,128,256,128,64,32)],
                'clf__learning_rate_init':[0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
            }
        }
    }
    figures = []
    Skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    t=[]
    for name, config in models.items():
        pipeline = config['pipeline']
        print(pipeline)
        params = config['params']
        
        for param_name, param_range in params.items():
            pipeline = config['pipeline']
            train_scores, test_scores = validation_curve(
                pipeline, X_train, y_train, param_name=param_name, param_range=param_range,
                cv=Skfold, scoring='f1_weighted', n_jobs=-1
            )
            pipeline = config['pipeline'] #get default pipeline for confusion matrix 
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            highest_test_score = np.max(test_mean)
            #get param value that corresponds to the highest test score
            highest_test_score_index = np.argmax(test_mean)
            highest_test_score_value = param_range[highest_test_score_index]
            t.append(f"{name} - {param_name}: {highest_test_score_value} - {highest_test_score:.4f}")
            #get param value that corresponds to the highest test score
            highest_test_score_index = np.argmax(test_mean)
            highest_test_score_value = param_range[highest_test_score_index]
            pipeline.set_params(**{param_name: highest_test_score_value})
            pipeline.fit(X_train,y_train)
            y_pred = pipeline.predict(X_test)
            from sklearn.metrics import f1_score
            f1 = f1_score(y_test, y_pred, average='weighted')
            print(f"{name} - {param_name}: {highest_test_score_value} - {highest_test_score:.4f} - {f1:.4f}")
            fig,ax=plt.subplots(figsize=(8, 6))
            if param_name == 'clf__penalty' or param_name == 'clf__solver':
                #bar chart
                ax.bar(param_range, test_mean, alpha=0.5, color='blue')
                ax.set_title(f"{name} - {param_name.replace('clf__', '').replace('_', ' ').title()}")
                ax.set_xlabel(param_name.replace('clf__', '').replace('_', ' ').title())
                ax.set_ylabel("F1 Weighted Score")
                ax.set_ylim(min(min(test_mean),min(train_mean))-0.08,max(max(test_mean),max(train_mean))+0.03)
                ax.grid(True)
                ax.annotate(f"Highest Test Score: {highest_test_score:.4f}, value: {highest_test_score_value}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', color="red")
                figures.append(fig)  # Add the figure to the list
                continue
            
            param_labels = param_range
            ax.set_title(f"{name} - {param_name.replace('clf__', '').replace('_', ' ').title()}")
            ax.set_xlabel(param_name.replace('clf__', '').replace('_', ' ').title())
            try:
                #3 significant digits
                if highest_test_score_value.round(3) == 0.000:
                    param_value = f'{highest_test_score_value:.5f}'
                else:
                    param_value = f"{highest_test_score_value.round(3)}"
            except:
                param_value = str(highest_test_score_value)
            if param_name == 'clf__var_smoothing':
                param_labels = [f"{x:.0e}" for x in param_range]
            elif param_name == 'clf__hidden_layer_sizes':
                print(param_range)
                temp = []
                for x in param_range:
                    print(x)
                    i = str(x)
                    i = i.replace(',','\n')
                    print(i)
                    temp.append(i)
                param_labels = temp
                #change figure size to 10,10
                fig.set_size_inches(10, 10)
            elif param_name == 'clf__priors':
                param_labels = ['None' if x is None else f'Prior {i+1}' for i, x in enumerate(param_range)]
            else:
                param_labels = param_range
            ax.set_ylabel("F1 Weighted Score")
            print(test_mean,train_mean)
            ax.set_ylim(min(min(test_mean),min(train_mean))-0.1,max(max(test_mean),max(train_mean))+0.1)
            lw = 2
            ax.plot(param_labels, train_mean, label="Training score", color="darkorange", lw=lw)
            ax.fill_between(param_labels, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange", lw=lw)
            ax.plot(param_labels, test_mean, label="Cross-validation score", color="navy", lw=lw)
            ax.fill_between(param_labels, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy", lw=lw)
            ax.legend(loc="best")
            ax.grid(True)
            ax.annotate(f"Highest Validation Score: {highest_test_score:.4f}, value: {param_value}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', color="red")            
            figures.append(fig)
            #plot confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            fig,ax=plt.subplots(figsize=(8, 6))
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{name} - {param_name.replace('clf__', '').replace('_', ' ').title()}({param_value}) - Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            figures.append(fig)
    plots_to_pdf.to_pdf(figures, filename=filename)

if __name__ == "__main__":
    import datetime
    filename = datetime.datetime.now().strftime("student_HyperTuning_%Y-%m-%d_%H-%M-%S.pdf")
    evaluate_parameters(filename=filename)
