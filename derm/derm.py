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
sys.path.append(os.path.abspath('.'))
import plots_to_pdf
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder

def evaluate_parameters(filename):
    import seaborn as sns
    try:
        df = pd.read_csv(os.path.abspath('../derm/dermatology_database_1.csv'))
    except:
        df = pd.read_csv(os.path.abspath('derm/dermatology_database_1.csv'))
    np.random.seed(42)
    df = df.dropna()
    df['age'] = df['age'].replace('?', np.nan)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(df[['age']])
    df['age'] = imputer.transform(df[['age']]).ravel().astype('int64')

     #correlation graph
    # matplotlib.use('TkAgg')
    # plt.figure(figsize=(18, 18))
    # sns.heatmap(df.corr(),annot=True, cmap='coolwarm',fmt='.2f')
    # plt.tight_layout()
    # plt.show()
    # exit()
    
    oenc = OrdinalEncoder()
    df_encoded = df.copy()
    df_encoded[df.columns] = oenc.fit_transform(df[df.columns])
    label_encoder = LabelEncoder()
    df_encoded['class'] = label_encoder.fit_transform(df['class'])
    print(df_encoded.head())
    X = df_encoded.drop(columns=['class'])
    y = df_encoded['class']
    print(y.value_counts())
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
    
    models ={
        'Naive Bayes': {
            'pipeline': Pipeline([('clf',nb.GaussianNB())]),
            'params':{
                'clf__priors':[None,[0.05,0.1,0.15,0.2,0.25,0.25]],
                'clf__var_smoothing':[1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
            }
        },
        'SVM': {
            'pipeline': Pipeline([('clf',svm.SVC(random_state=42))]),
            'params':{
                'clf__C':[0.001,0.01,0.1,1,10,100],
                'clf__kernel':['linear','poly','rbf','sigmoid']
            }
        },
        'Neural Network': {
            'pipeline': Pipeline([('clf',nn.MLPClassifier(random_state=42))]),
            'params':{
                'clf__hidden_layer_sizes':[(100,),(16,32,32,16),(20,30,20),(30,40,30),(40,50,40),(50,60,50)],
                # 'clf__learning_rate_init':[0.0001,0.0005,0.0007,0.001,0.0015,0.002,0.01,0.1,0.12]
                'clf__learning_rate_init' : np.linspace(0.0001,0.1,100)
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
                cv=Skfold, scoring='recall_macro', n_jobs=-1
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
            from sklearn.metrics import recall_score
            recall = recall_score(y_test, y_pred, average='macro')
            print(f"{name} - {param_name}: {highest_test_score_value} - {highest_test_score:.4f} - {recall:.4f}")
            fig,ax=plt.subplots(figsize=(8, 6))
            
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
                for x in range(len(param_range)):
                    temp.append(x)
                param_labels = temp
                # Add colored dots for each label's data points
                colors = plt.cm.rainbow(np.linspace(0, 1, len(param_range)))
                for i, (label, color) in enumerate(zip(param_range, colors)):
                    ax.plot(i, test_mean[i], 'o', color=color, markersize=8, label=label, zorder=3)
                    ax.plot(i, train_mean[i], 'o', color=color, markersize=8, zorder=3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                ax.set_xlabel("Hidden layer index")
                ax.set_xticks(range(len(param_labels)))
                ax.set_xticklabels(param_labels)
            elif param_name == 'clf__priors':
                temp = []
                for x in param_range:
                    if x is None:
                        temp.append('None')
                    else:
                        i = str(x)
                        i = i.replace(',','\n')
                        temp.append(i)
                param_labels = temp
                fig.set_size_inches(8,6)
                #horizontal bar chart
                ax.barh(param_labels, test_mean,0.5, alpha=0.5, color='blue')
                ax.set_title(f"{name} - {param_name.replace('clf__', '').replace('_', ' ').title()}")
                ax.set_ylabel(param_name.replace('clf__', '').replace('_', ' ').title())
                ax.set_xlabel("Recall Score")
                ax.set_xlim((min(test_mean))-0.005,max(test_mean)+0.005)
                ax.set_ylim(top=2)
                ax.grid(True)
                ax.annotate(f"Highest Test Score: {highest_test_score:.4f}, value: {highest_test_score_value}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', color="red")
                figures.append(fig)  # Add the figure to the list
                continue
            else:
                param_labels = param_range
            ax.set_ylabel("Recall Score")
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
    filename = datetime.datetime.now().strftime("../derm/dermatology_HyperTuning_%Y-%m-%d_%H-%M-%S.pdf")
    evaluate_parameters(filename=filename)
