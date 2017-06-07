import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.metrics import roc_auc_score as AUC
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from scipy.stats.mstats import gmean
from sklearn.cross_validation import train_test_split as split
from copy import copy
os.chdir('/Users/adarshchavakula/Documents/Kaggle/Santander')

class XGC:
    '''
    Class to create a similar-to-Scikit-learn API for XG Boost Classifier object. 
    
    Initialize: 
    booster = XGC(params=param_dict, num_rounds=100)
    
    Methods:
    fit(x,y)
        fit the XGBoost model on feature array x and labels y.

    predict_proba(test)
        Make predictions for new test data based on what the model learned in the fit procedure. 

    '''
    def __init__(self,params,num_rounds):
        self.params = params
        self.num_rounds = num_rounds
        return
    
    def fit(self,x,y):
        xgtrain = xgb.DMatrix(x, label=y,missing=np.nan)
        watchlist = [(xgtrain, 'train')]
        self.model = xgb.train(list(self.params.items()), xgtrain, self.num_rounds, watchlist,verbose_eval=False)
        return
        
    def predict_proba(self,test):
        xgtest = xgb.DMatrix(test,missing=np.nan)
        probs = self.model.predict(xgtest)
        probs_sk = np.zeros([len(probs),2])
        probs_sk[:,1]=probs
        return np.array(probs_sk)

def duplicate_columns(frame):
    '''
    function to dump the duplicate columns in the data
    '''
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []
    for t, v in groups.items():
        dcols = frame[v].to_dict(orient="list")

        vs = dcols.values()
        ks = dcols.keys()
        lvs = len(vs)

        for i in range(lvs):
            for j in range(i+1,lvs):
                if vs[i] == vs[j]: 
                    dups.append(ks[i])
                    break
    return dups

def constant_columns(frame):
    '''
    function to dump the columns which do not vary at all. 
    It is not necessary to remove constant columns for tree based methods but 
    they can add unncessary bias to distance based methods or neural networks
    '''
    remove = []
    for col in frame.columns:
        if frame[col].std() == 0:
            remove.append(col)
            return remove

class bagger:
    '''
    Class to create a bagger object which does Boostrap Aggregation (Bagging) of any chosen model. 
    The bagger object has an API similar to scikit learn models and can be used in a similar fashion.

    Initialize: 
    bag = bagger(clf,num_bags=100,bag_fraction=0.8)
    where clf is any scikit learn model (example clf = sklearn.ensemble.GradientBoostingClassifier) or an XGC object.
    bag_fraction = decides what percentage of samples must be selected for each bag.
    num_bags = number of bags.
    
    Methods:
    fit(x,y)
        fit the bagger model on feature array x and labels y.

    predict_proba(test)
        Make predictions for new test data based on what the model learned in the fit procedure. 
    '''
    def __init__(self,clf,num_bags=10,bag_fraction=0.66):
        self.clf=clf
        self.num_bags=num_bags
        self.bag_fraction = bag_fraction
        return
    def fit(self,x,y):
        trained_models=[]
        for bag in range(self.num_bags):
            x,y = shuffle(x,y,random_state=bag*3)
            #xtrain,xtest, ytrain,ytest = split(x, y, test_size=1.0-self.bag_fraction, stratify=y,random_state=42*bag)
            xtrain,xtest, ytrain,ytest = split(x, y, test_size=1.0-self.bag_fraction,random_state=42*bag)
            mod = copy(self.clf)
            mod.fit(x,y)
            trained_models.append(mod)
        self.trained_models=trained_models
    def predict_proba(self,x):
        preds = np.zeros((len(x),self.num_bags))
        for n,mod in enumerate(self.trained_models):
            preds[:,n]=mod.predict_proba(x)[:,1]
        avg_pred = gmean(preds,axis=1)
        probs_sk = np.zeros([len(avg_pred),2])
        probs_sk[:,1]=np.ravel(avg_pred) # probs in sklearn predict proba format
        return np.array(probs_sk)



def readData():
    '''
    Function to read the datasets and clean them up.
    '''
    print 'Reading Data...'
    x = pd.read_csv('train.csv')
    xt = pd.read_csv('test.csv')
    sample = pd.read_csv('sample_submission.csv')
    y = np.array(x['TARGET'])
    x.drop(['ID','TARGET'],axis=1,inplace=True)
    xt.drop(['ID'],axis=1,inplace=True)
    # Remove duplocate columns and constant columns
    dups = duplicate_columns(x)
    x.drop(dups,axis=1,inplace=True)
    xt.drop(dups,axis=1,inplace=True)
    cons = duplicate_columns(x)
    x.drop(cons,axis=1,inplace=True)
    xt.drop(cons,axis=1,inplace=True)

    # consider the most important variables and use Naive Bayes to calculate a probable score and add to the dataset as a predictor
    #x_imp = x[['var15','ind_var30','ind_var5','var38','num_meses_var5_ult3','var36']]
    #xt_imp = xt[['var15','ind_var30','ind_var5','var38','num_meses_var5_ult3','var36']]
    #bayes = NB() #NAIVE BAYES
    #bayes.fit(np.array(x_imp),y)
    #x['bayes']=bayes.predict_proba(np.array(x_imp))[:,1]
    #xt['bayes']=bayes.predict_proba(np.array(xt_imp))[:,1]

    # Calculate some derived variables based on their definitions 
    xt['nv1']=xt['num_var33']+xt['saldo_medio_var33_ult3']+xt['saldo_medio_var44_hace2']+xt['saldo_medio_var44_hace3']+xt['saldo_medio_var33_ult1']+xt['saldo_medio_var44_ult1']
    x['nv1']=x['num_var33']+x['saldo_medio_var33_ult3']+x['saldo_medio_var44_hace2']+x['saldo_medio_var44_hace3']+x['saldo_medio_var33_ult1']+x['saldo_medio_var44_ult1']

    # DROP BAD FEATURES
    bad_feats=['ind_var18_0', 'ind_var33_0', 'ind_var33', 'ind_var34', 'num_var13_medio', 'num_var18', 'num_op_var40_hace3', 'num_var29_0', 'num_var29', 
               'num_var33', 'num_var34_0', 'saldo_var6', 'saldo_var13_medio', 'saldo_var18', 'saldo_var20', 'saldo_var33', 'saldo_var34', 'delta_imp_amort_var18_1y3', 'delta_imp_amort_var34_1y3', 
               'delta_imp_aport_var33_1y3', 'delta_imp_trasp_var17_in_1y3', 'delta_imp_trasp_var17_out_1y3', 'delta_imp_trasp_var33_in_1y3', 'delta_imp_trasp_var33_out_1y3', 'delta_imp_venta_var44_1y3', 
               'delta_num_aport_var33_1y3', 'delta_num_reemb_var33_1y3', 'delta_num_venta_var44_1y3', 'imp_amort_var18_ult1', 'imp_amort_var34_ult1', 'imp_aport_var17_hace3', 'imp_aport_var33_hace3', 
               'imp_aport_var33_ult1', 'imp_var7_emit_ult1', 'imp_compra_var44_hace3', 'imp_reemb_var17_hace3', 'imp_reemb_var33_ult1', 'imp_trasp_var17_in_hace3', 'imp_trasp_var17_in_ult1', 
               'imp_trasp_var17_out_ult1', 'imp_trasp_var33_in_hace3', 'imp_trasp_var33_in_ult1', 'imp_trasp_var33_out_ult1', 'imp_venta_var44_hace3', 'imp_venta_var44_ult1', 'ind_var7_emit_ult1', 
               'num_aport_var17_hace3', 'num_aport_var33_hace3', 'num_aport_var33_ult1', 'num_var7_emit_ult1', 'num_compra_var44_hace3', 'num_meses_var13_medio_ult3', 'num_meses_var29_ult3', 
               'num_meses_var33_ult3', 'num_reemb_var13_hace3', 'num_reemb_var17_hace3', 'num_reemb_var33_ult1', 'num_trasp_var17_in_hace3', 'num_trasp_var17_in_ult1', 'num_trasp_var17_out_ult1', 
               'num_trasp_var33_in_hace3', 'num_trasp_var33_in_ult1', 'num_trasp_var33_out_ult1', 'num_venta_var44_hace3', 'num_venta_var44_ult1']

    x.drop(bad_feats,axis=1,inplace=True)
    xt.drop(bad_feats,axis=1,inplace=True)
    
    # Concatenate the training and testing datasets - need it for a couple of purposes below
    c = pd.concat((x,xt))

    # Digitize the var15 variable into buckets, take logarithm of var38
    c['AGE']=np.digitize(np.array(c['var15']),np.linspace(20,100,9))
    c['LOG38']=np.digitize(np.log10(np.array(c['var38'])),np.linspace(3,7,9))
    c['var38']=np.log10(np.array(c['var38'])+1)
    #c = pd.get_dummies(c,columns=['AGE','LOG38'])
    #c.drop(['var15','var38'],axis=1,inplace=True)
    feats = list(c)
    x = c.iloc[:len(x)]
    xt = c.iloc[len(x):]

    # Do some PCA and add those principal components to the datasets - just 3 components - adds some predictive value
    x = np.array(x)
    xt = np.array(xt)
    princomp = PCA(n_components=3,whiten=True)
    princomp.fit(x)
    x_pc = princomp.transform(x)
    xt_pc = princomp.transform(xt)
    x = np.append(x,x_pc,axis=1)
    xt = np.append(xt,xt_pc,axis=1)
    xmin,xmax = np.min(x,axis=0),np.max(x,axis=0)

    # ensure that no variables in the test data exceed the bounds of the training data
    for i in range(np.shape(xt)[1]):
        xt[xt[:,i]>xmax[i],i]=xmax[i]
        xt[xt[:,i]<xmin[i],i]=xmin[i]
    
    return x,y,xt,feats,sample
    
def crossValidate(clf,x,y,folds=5,runs=5):
    '''
    Function for doing K-Fold cross validation.
    clf = classifier
    x = training data, numpy NDarray
    y = labels, numpy 0D array
    folds = number of partitions to be made for the training data
    runs = number of times to repeat the cross validation process, each time with a different random partition.

    folds=5 and runs=10 will do a 5-fold cross validation 10 times on the dataset and calculate the AUC deviation accross these 50 instances.
    '''
    ypred = np.zeros((len(y),runs))
    fold_auc = np.zeros((runs,folds))
    r=0
    score = np.zeros(runs)
    for run in range(runs):
        i=0
        x,y = shuffle(x,y,random_state=19*(run+3)) # some random seeding to be unique
        kf = KFold(y,n_folds=folds,random_state=18*(run+93))
        print 'Cross Validating...'
        for train_ind,test_ind in kf:
            print 'CV Fold ' + str(i+1) + ' out of ' + str(folds)
            xtrain,ytrain = x[train_ind,:],y[train_ind]
            xtest,ytest = x[test_ind,:],y[test_ind]
            clf.fit(xtrain,ytrain)
            #a =  100*clf.feature_importances_
            #print ["%0.3f" % f for f in a]
            fold_pred = clf.predict_proba(xtest)[:,1]
            fold_pred[xtest[:,1]<23]=0
            fold_auc[r,i] = AUC(ytest,fold_pred)
            ypred[test_ind,r]=fold_pred
            i=i+1
        score[r] = AUC(y,ypred[:,r])
        r=r+1
    print 'Fold AUC: ' + str(fold_auc)
    print 'Mean: ' + str(np.mean(fold_auc))
    print 'Deviation: ' + str(np.std(fold_auc))
    
    print '\nOverall AUC: '+ str(score)
    print 'Mean: ' + str(np.mean(score))
    print 'Deviation: ' + str(np.std(score))
    return score
        
def makeSubmission(clf,train,y,test,sample,filename):
    '''
    Function to make submissions for a test datset and store them into a chosen csv file
    classifier is trained on the train and y and predictions are made for test.
    '''
    print 'Training classifier...'
    clf.fit(train,y)
    preds = clf.predict_proba(test)[:,1]
    preds[test[:,1]<23]=0
    sample['TARGET']= preds
    print 'Writing predictions to submission file...'
    sample.to_csv(filename,sep=',',index=False)
    return



def main():
    # Get the clean datasets
    x,y,xt,feats,sample = readData()

    #Try out different models
    xg_class_params = {"objective" : "binary:logistic","eval_metric" : "auc", "booster" : "gbtree",
                       "eta": 0.01,"max_depth": 14,"min_child_weight": 10,                    
                       "subsample": 0.66,
                       #"colsample_bytree": 0.7,
                       "colsample_bylevel":0.3,                       
                       "thread": 1,"silent": 1,"seed": 221}
    xg_class_params2 = {"objective" : "binary:logistic","eval_metric" : "auc", "booster" : "gbtree",
                       "eta": 0.02,"max_depth": 5,"min_child_weight": 10,                    
                       "subsample": 0.66,
                       #"colsample_bytree": 0.7,
                       "colsample_bylevel":0.3,                       
                       "thread": 1,"silent": 1,"seed": 221}
    rf1 = RF(n_estimators=1000,max_features= 50,criterion='entropy',min_samples_split= 40,max_depth= 30, min_samples_leaf= 2, n_jobs = 10,verbose=0,random_state=42)
    etc1 = ETC(n_estimators=500,max_features= 90,criterion='entropy',min_samples_split= 20,max_depth= 25, min_samples_leaf= 10, n_jobs =10,verbose=0,random_state=42)
    xgb1 = XGC(xg_class_params,num_rounds=550)
    xgb2 = XGC(xg_class_params2,num_rounds=600)
    xgb_bag=bagger(xgb2,num_bags=3,bag_fraction=0.75)

    # EVALUATE a model
    score = crossValidate(etc1,x,y,folds=5,runs=1)

    # Make a prediction
    #makeSubmission(xgb_bag,x,y,xt,sample,'submission_XGB10.csv')
    

if __name__ == "__main__":
    main()