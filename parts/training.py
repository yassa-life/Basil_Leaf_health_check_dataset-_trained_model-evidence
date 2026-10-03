"""End-to-end training with fold-local preprocessing and model-specific tuning."""
from pathlib import Path
import argparse, hashlib, json, time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from threadpoolctl import threadpool_limits
from parts._pipeline import prepare_dataset, CLASSES, SEED, FEATURE_VERSION
from parts.evaluation import evaluate_selected, save_main

ROOT = Path(__file__).resolve().parents[1]
NAMES = dict(logistic_regression='Logistic Regression', svm='SVM', knn='KNN', decision_tree='Decision Tree', random_forest='Random Forest', gradient_boosting='Gradient Boosting')
RATIONALE = {
 'logistic_regression': 'Regularized linear boundary; C controls complexity with high-dimensional features.',
 'svm': 'RBF kernel captures nonlinear boundaries; C and gamma control complexity and locality.',
 'knn': 'Local similarity baseline; neighbors and distance weights control smoothing.',
 'decision_tree': 'Interpretable nonlinear rules; depth and leaf size limit overfitting.',
 'random_forest': 'Averaged randomized trees reduce variance; leaf size and feature sampling regulate complexity.',
 'gradient_boosting': 'Sequential trees correct residual errors; learning rate and depth regulate complexity.'}

def specification(key):
    specs = {
      'logistic_regression': (LogisticRegression(max_iter=4000, class_weight='balanced', random_state=SEED), {'C':[0.01,0.1,1,10]}),
      'svm': (SVC(class_weight='balanced', random_state=SEED), {'C':[1,10], 'gamma':['scale',0.001]}),
      'knn': (KNeighborsClassifier(), {'n_neighbors':[3,5,11], 'weights':['uniform','distance']}),
      'decision_tree': (DecisionTreeClassifier(class_weight='balanced', random_state=SEED), {'max_depth':[3,6,None], 'min_samples_leaf':[2,5]}),
      'random_forest': (RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=2), {'min_samples_leaf':[1,3], 'max_features':['sqrt',0.3]}),
      'gradient_boosting': (GradientBoostingClassifier(n_estimators=100, max_features='sqrt', random_state=SEED), {'learning_rate':[0.05,0.1], 'max_depth':[2,3]})}
    estimator, params = specs[key]
    pipe = Pipeline([('scale',StandardScaler()), ('select',SelectPercentile(f_classif)), ('model',estimator)])
    grid = {'select__percentile':[25,100], **{'model__'+k:v for k,v in params.items()}}
    return pipe, grid

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')

def train_model(key, data):
    out = ROOT/'parts'/key/'outputs'
    out.mkdir(parents=True, exist_ok=True)
    pipe, grid = specification(key)
    search = GridSearchCV(pipe, grid, scoring={'f1':'f1_macro','accuracy':'accuracy','balanced_accuracy':'balanced_accuracy'}, refit='f1', cv=data['cv'], n_jobs=1, return_train_score=True, error_score='raise')
    print(f'Tuning {NAMES[key]}: {grid}', flush=True)
    start=time.perf_counter()
    with threadpool_limits(limits=2): search.fit(data['X_tr'],data['y_tr'])
    duration=time.perf_counter()-start
    results=pd.DataFrame(search.cv_results_)
    results.to_csv(out/'tuning_results.csv',index=False)
    results.groupby('param_select__percentile')[['mean_test_f1','mean_test_accuracy']].max().to_csv(out/'preprocessing_comparison.csv')
    fingerprint=hashlib.sha256(data['frame'][['path','label','sha256','split_group']].to_csv(index=False).encode()).hexdigest()
    bundle=dict(estimator=search.best_estimator_, classes=CLASSES, name=NAMES[key], feature='handcrafted', feature_version=FEATURE_VERSION, seed=SEED, dataset_fingerprint=fingerprint)
    metrics,pred=evaluate_selected(bundle,data['X'],data['frame'],data['test'])
    i=search.best_index_
    metrics.update(model_name=NAMES[key], best_params=search.best_params_, cv_macro_f1=search.best_score_, cv_std=float(results.iloc[i].std_test_f1), cv_accuracy=float(results.iloc[i].mean_test_accuracy), mean_fit_seconds=float(results.iloc[i].mean_fit_time), tuning_seconds=duration, refit_seconds=search.refit_time_, candidate_count=len(results), cv_fits=len(results)*len(data['cv']), n_train=len(data['dev']), n_test=len(data['test']), dataset_fingerprint=fingerprint, rationale=RATIONALE[key], selection='Highest grouped development CV macro-F1; test excluded.')
    save_json(out/f'{key}_metrics.json',metrics)
    save_json(out/'search_space.json',grid)
    joblib.dump(bundle,out/f'{key}_model.joblib')
    predictions=data['frame'].iloc[data['test']][['path','label','split_group']].copy()
    predictions['prediction']=pred
    predictions.to_csv(out/'test_predictions.csv',index=False)
    data['manifest'].to_csv(out/'split_manifest.csv',index=False)
    save_json(out/'data_audit.json',data['audit'])
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(results.mean_train_f1.to_numpy(),label='Training')
    axes[0].errorbar(range(len(results)),results.mean_test_f1,yerr=results.std_test_f1,label='Grouped validation',fmt='o-')
    axes[0].set(xlabel='Candidate index (tuning_results.csv)',ylabel='Macro-F1',title=NAMES[key]+' tuning'); axes[0].legend()
    cm=np.array(metrics['confusion_matrix']); axes[1].imshow(cm,cmap='Blues')
    axes[1].set(xticks=[0,1],yticks=[0,1],xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='Actual',title='Holdout confusion matrix')
    for a in range(2):
        for b in range(2): axes[1].text(b,a,str(cm[a,b]),ha='center',va='center',color='red')
    fig.tight_layout(); fig.savefig(out/'evaluation.png',dpi=160); plt.close(fig)
    print(f'{NAMES[key]}: CV F1 {search.best_score_:.4f}; holdout F1 {metrics["macro_f1"]:.4f}',flush=True)
    return metrics,bundle

def run(keys=None):
    data=prepare_dataset(ROOT)
    keys=keys or list(NAMES)
    rows=[]; estimators={}
    for key in keys:
        m,b=train_model(key,data)
        rows.append(dict(model=NAMES[key],features='handcrafted',cv_macro_f1=m['cv_macro_f1'],cv_std=m['cv_std'],cv_accuracy=m['cv_accuracy'],mean_fit_seconds=m['mean_fit_seconds'],accuracy=m['accuracy'],macro_f1=m['macro_f1'],tuning_seconds=m['tuning_seconds'],candidate_count=m['candidate_count'],fold_f1=[]))
        estimators[NAMES[key]]=b['estimator']
    if set(keys)==set(NAMES):
        comparison=pd.DataFrame(rows).sort_values(['cv_macro_f1','mean_fit_seconds','model'],ascending=[False,True,True]).reset_index(drop=True)
        save_main(data,comparison,estimators)
        comparison.drop(columns='fold_f1').to_csv(ROOT/'outputs'/'six_model_comparison.csv',index=False)
    return data,rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(NAMES))
    args=parser.parse_args()
    run([args.model] if args.model else None)

if __name__=='__main__': main()
