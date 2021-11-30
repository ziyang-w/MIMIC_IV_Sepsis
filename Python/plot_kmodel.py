import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import datetime
from tqdm import tqdm
import json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, auc
from sklearn.metrics import roc_curve

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


def del_missing(file_path,
                file_name,
                outName,
                missing_rate=0.1,
                saveFlag=True):
    import os
    import pandas as pd
    '''
    1. load data from "file_path/file_name"
    2. fill_na ventilation with 0
    3. delete features which missing_rate > missing_rate
    4. saveFlag = Ture, save csv ,and return  file_path/file_name
                = False, return datOUT

    :param missing_rate: float,threshold
    :param file_path: 
    :param file_name:
    :param outName:
    :return: out_file name path
    '''
    csv = os.path.join(file_path, file_name)
    datIN = pd.read_csv(csv, index_col=0)
    print('datIN.shape: %s' % str(datIN.shape))
    if 'vent_status' in datIN.columns.tolist():
        datIN.vent_status.fillna(0)
    features = (datIN.isnull().sum() / len(datIN)) < missing_rate
    features.tolist()

    datOUT = datIN.loc[:, features.tolist()]

    # 删除有缺失值的行
    datOUT.dropna(inplace=True)
    print('datOUT.shape: %s' % str(datOUT.shape))

    if outName == '':
        outName = file_name.split('.')[0] + '_del_missing.csv'
    print('saving %s.csv' % outName)

    if saveFlag==True:
        datOUT.to_csv(os.path.join(file_path, outName))
        return os.path.join(file_path, outName)
    return datOUT


def read_data(file,
              file_path='',
              file_name='',
              index_colol=None):
    '''
    1. load data from file, you can also use file_path,file_name to get fileIO
    2. return DF, which has deleted the columns contained 'id', 'time' words
    :param file:
    :param file_path:
    :param file_name:
    :param index_colol:
    :return:
    '''
    if file == '':
        print('no file pass in! ')
    if file_name[-4:] == '.csv':
        dat = pd.read_csv(file)
    if file_name[-4:] == 'xlsx':
        dat = pd.read_excel(file)

    print('readIN_shape %s' % str(dat.shape))

    dropTime = [x for x in dat.columns if 'time' in x]
    dropId = [x for x in dat.columns if '_id' in x]

    dat.drop(dropTime, axis=1, inplace=True)
    print('drop_Time_shape %s' % str(dat.shape))
    dat.drop(dropId, axis=1, inplace=True)
    print('drop_ID_shape %s' % str(dat.shape))

    return dat


def missing_dect(df, missingRate=0.1, method='mode'):
    '''统计每个变量的缺失数据的数量，编量名称的DataFrame'''
    cols = {
        'feature_no': [],
        'Variables': [],
        'null_num': [],
        'null_rate(%)': []
    }
    for e, c in enumerate(df.columns):
        cols['feature_no'].append(e)
        cols['Variables'].append(c)
        cols['null_num'].append(sum(pd.isnull(dat[c])))
        cols['null_rate(%)'].append(sum(pd.isnull(df[c])) / len(df[c]))
    datMS = pd.DataFrame(cols)  # .sort_values(ascending=False,by='null_num')

    dat_data = dat.loc[:, list(datMS['null_rate(%)'] < missingRate)]
    print('输入的形状%s' % str(dat_data.shape))
    print('==下面的特征因数据缺失率>%f而被删除==' % missingRate)
    print(datMS[datMS['null_rate(%)'] > missingRate]['Variables'])

    # 缺失值填补，
    if method == 'mode':
        # 采用众数填补
        print('=' * 20, '填补的缺失值数量', '=' * 20)
        for item in dat_data.columns:
            if dat_data.loc[:, item].isna().sum() > 0:
                print(item, dat_data.loc[:, item].isna().sum())
                dat_data.loc[:, item] = dat_data.loc[:, item].fillna(
                    dat_data.loc[:, item].mode()[0])  # 采用第一个众数填补缺失值
    return dat_data


def Boxing_Numeric(dat_data, cut=4):
    # 分箱
    DFBoxed = dat_data.copy()
    for var in DFBoxed.columns.values:
        if type(DFBoxed[var][0]) != str:
            if DFBoxed[var].mode()[0] not in [0, 1, 2, 3, 4,5,
                                              100]:  # 众数不为1,0，说明为连续变量
                DFBoxed[var] = pd.qcut(DFBoxed[var].tolist(),
                                       q=cut,
                                       labels=[x for x in range(cut)])  # 按照4分位数切分

    # 变量数值化
    dat_data = DFBoxed
    ch_dict = []
    for item in dat_data.columns:
        if type(dat_data[item][1]) == str:
            # 字符型变量，进行数值化操作
            unique_value = dat_data[item].unique().tolist()
            unique_value.sort()
            temp = [item]
            for k, v in enumerate(unique_value):
                temp.append([v, k])
            ch_dict.append(temp)
            dat_data[item] = dat_data[item].apply(
                lambda x: unique_value.index(x))
    #     print('=' * 20, '特征编码方案', '=' * 20)
    #     print(pd.DataFrame(ch_dict))
    pd.DataFrame(ch_dict).to_csv(
        os.path.join(log_path, hour) + 'encoding.csv')

    return dat_data


# def lasso_selection(df,k):
#     #带L1 lasso回归
#     xDF = df[df.columns.difference([targetY])]
#     yDF = df[targetY]
#
#     clf=SelectFromModel(LogisticRegression(penalty='l1',C=1000,solver='liblinear') )#SelectFromModel
#     M=clf.fit(xDF,yDF)
#
#     feature=xDF.loc[:,list(M.get_support())]
#     # todo: Lasso怎么返回分数
#     result=pd.DataFrame({'var':xDF.columns.tolist(),'Lasso':}).sort_values(by='MIC',ascending=False)
#     result.reset_index().to_csv(os.path.join(log_path,hour+'MIC.csv'))
#     xDF[result.iloc[:k, 0].tolist()].join(yDF).to_csv(os.path.join(log_path,hour+'MIC_selection.csv'))
#     return xDF[result.iloc[:k,0].tolist()], yDF


def MIC_selection(df,k=20):
    from sklearn.feature_selection import mutual_info_classif as MIC
    xDF = dat_data[dat_data.columns.difference([targetY])]
    yDF = dat_data[targetY]
    result=pd.DataFrame({'var':xDF.columns.tolist(),'MIC':MIC(xDF, yDF)}).sort_values(by='MIC',ascending=False)
    result.reset_index().to_csv(os.path.join(log_path,hour+'MIC.csv'))
    xDF[result.iloc[:k, 0].tolist()].join(yDF).to_csv(os.path.join(log_path,hour+'MIC_selection.csv'))
    return xDF[result.iloc[:k,0].tolist()], yDF

def LR_kfold(xDF, yDF, k=2, random_state=0):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    # 对其中1折进行预测，对其他折进行训练
    skfolds = StratifiedKFold(n_splits=k,
                              shuffle=True,
                              random_state=random_state)

    model = LogisticRegression(max_iter=1000).fit(np.array(xDF), np.array(yDF))

    cols = {'P': [], 'R': [], 'F1': [], 'AUC': []}
    # for model in modelList:
    for train_index, test_index in skfolds.split(xDF, yDF):
        X_train_folds = xDF.iloc[train_index, :]
        y_train_folds = yDF.iloc[train_index, :]
        X_test_fold = xDF.iloc[test_index, :]
        y_test_fold = yDF.iloc[test_index, :]

        P, R, F1, AUC = LR(X_train_folds, yDF)

        ypre = model.predict(np.array(xtest))
        yprob = model.predict_proba(np.array(xtest))

        P, R, F1, AUC = PRF1(np.array(ytest), ypre, yprob)
        print(P, R, F1, AUC)

        clone_clf.fit(X_train_folds, y_train_folds)
        # 预测测试集中数据的分类类别
        y_pred = clone_clf.predict(X_test_fold)
        y_prob = clone_clf.predict_proba(X_test_fold)
        # 将预测得到的分类信息与测试集中原有的分类信息进行比较，求出相同的数量的和
        n_correct = sum(y_pred == y_test_fold)
        p, r, f1, auc = PRF1(Y_test_fold, y_pre, y_prob)
        cols['P'].append(p)
        cols['R'].append(r)
        cols['F1'].append(f1)
        cols['AUC'].append(auc)

        # 将该和除以总共预测的值的数量，计算准确率
        print(n_correct / len(y_pred))


def Bmodel(xtrain, ytrain, idnum):
    # 构造数据集合
    dataTrain = xtrain.join(ytrain)

    # 结构学习
    hc = HillClimbSearch(dataTrain)
    best_model = hc.estimate(scoring_method=BicScore(dataTrain))

    outName = os.path.join(log_path, hour + 'BN_Fold%d' % idnum)

    showBN(best_model, save=True, outName=outName)

    # 参数学习
    model = BayesianNetwork(best_model.edges())
    model.fit(data=dataTrain,
              estimator=BayesianEstimator,
              prior_type='BDeu',
              equivalent_sample_size=1000)

    print('dataTrain.shape: %s' % str(dataTrain.shape))

    return model, best_model


def Bpre(model, xtest, targetY):
    from pgmpy.inference import VariableElimination
    model_infer = VariableElimination(model)

    query_dict = {}
    Q = []
    Q_prob = []
    errorList = []

    for i, item in enumerate(np.array(xtest)):
        for var, feature in zip(item, xtest.columns.tolist()):
            if feature in list(model.nodes):
                query_dict[feature] = var
        print(i)
        #         q=model_infer.map_query(variables=[targetY],evidence=query_dict)
        #         q_prob=model_infer.query(variables=[targetY],evidence=query_dict)
        try:
            q = model_infer.map_query(variables=[targetY], evidence=query_dict)
            q_prob = model_infer.query(variables=[targetY],
                                       evidence=query_dict)
        except:
            errorList.append(i)
            print('第%d发生了错误，并且将其保存在了errorList中' % i)

        Q.append(q.get(targetY))
        Q_prob.append(q_prob.values[1])
    return Q, Q_prob, errorList


def RF(xtrain, ytrain, random_state=0):
    model = RandomForestClassifier(random_state=random_state)
    model.fit(xDF, yDF)

    ypre = model.predict(np.array(xtest))
    yprob = model.predict_proba(np.array(xtest))

    P, R, F1, AUC = PRF1(np.array(ytest), ypre, yprob)
    print(P, R, F1, AUC)
    return P, R, F1, AUC


def LR(xDF, yDF, random_state=0):
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(xDF, yDF)

    ypre = model.predict(np.array(xtest))
    yprob = model.predict_proba(np.array(xtest))

    P, R, F1, AUC = PRF1(np.array(ytest), ypre, yprob)
    print(P, R, F1, AUC)
    return P, R, F1, AUC


def showBN(model, outName, save=False):
    '''传入BayesianModel对象，调用graphviz绘制结构图，jupyter中可直接显示'''
    from graphviz import Digraph
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    edges = model.edges()
    for a, b in edges:
        dot.edge(a, b)
    if save:
        dot.view(outName, cleanup=True)
    return dot



def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def PRF1(ytest, ypre, yproba,threshold=0.5):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    if threshold == 0.5:
        myA= accuracy_score(ytest, ypre)
        myP = precision_score(ytest, ypre)
        myR = recall_score(ytest, ypre)
        F1 = f1_score(ytest, ypre)
        auc = roc_auc_score(ytest, yproba)
        return myP, myR, F1, auc
    else:
        myA = accuracy_score(ytest, [1 if x > threshold else 0 for x in yprob])
        myP = precision_score(ytest, [1 if x > threshold else 0 for x in yprob])
        myR = recall_score(ytest, [1 if x > threshold else 0 for x in yprob])
        F1 = f1_score(ytest, [1 if x > threshold else 0 for x in yprob])
        auc = roc_auc_score(ytest, yproba)
        return myA, myP, myR, F1, auc


def plot_ROC(yprob, ytest, l):
    from sklearn.metrics import roc_curve, auc

    FPR, recall, _ = roc_curve(ytest.ravel(), yprob.ravel())

    plt.plot(FPR, recall, 'r-', label=l + ', AUC=%.2f' % auc(FPR, recall))
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(l)
    plt.legend()
    plt.show()


def plot_ROC_kfold_my(fprs, tprs, l):
    aucs = []
    maxLen = max([len(x) for x in fprs])
    fig = plt.figure(figsize=(12, 9), dpi=150)
    i = 0

    for fpr, tpr in zip(fprs, tprs):
        dif = maxLen - len(fpr)
        fprs[i] = np.append(fpr, np.ones(dif))
        tprs[i] = np.append(tpr, np.ones(dif))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # 绘制每一折的ROC曲线
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i + 1, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    # 绘制平均ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_fpr = np.mean(fprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    # 绘制ROC曲线的上下界
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s: ROC' % l)
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(log_path, hour + l + '.pdf'), dpi=300)

    # plt.show()


def plot_ROC_kfold(tprs, opts, l):
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # mean_fpr = max()
    fig = plt.figure(figsize=(12, 9), dpi=150)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    i = 0

    for tpr, opt in zip(tprs,opts):
        roc_auc = auc(mean_fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(mean_fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f, th = %0.2f)' % (i + 1, roc_auc,opt[0]))
        plt.plot(opt[1][0], opt[1][1], marker='o', color='r', alpha=0.3)
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2,
             alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('假阳性率（False Positive Rate）')
    plt.ylabel('真阳性率（True Positive Rate）')
    plt.title('ROC: %s' % l)
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(log_path, hour + l + '.pdf'), dpi=300)

    # plt.show()


def plot_ROC_kmodel(kmodel):
    aucs = []
    tprs_mean = kmodel['TPR_MEAN']
    tprs_std = kmodel['TPR_STD']
    l = kmodel['L']
    mean_fpr = np.linspace(0, 1, 100)
    fig = plt.figure(figsize=(9, 9), dpi=150)
    i = 0

    for mean_tpr, std_tpr, l in zip(tprs_mean, tprs_std, l):
        roc_auc = auc(mean_fpr, mean_tpr)
        aucs.append(roc_auc)
        plt.plot(mean_fpr, mean_tpr, lw=1, label=l + '(AUC = %0.2f)' % roc_auc)

        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC' )
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(log_path, hour + 'ROC_kmodel.pdf'), dpi=300)

    plt.show()


############################################## 全局参数  ###########################################
if __name__ == '__main__':
    fileName = 'final_sepsis.csv'
    targetY = 'death_within_30_days'

    # fileName = sys.argv[1]
    # targetY = sys.argv[2]

    filePath = r'D:\Data\secret\MIMIC_MY'  # Windows_file_path
    # filePath=r'~/Documents/Data/secret/MIMIC_MY' #Mac_file_path

    missingRate = 0.1
    fold = 10
    # 可以优化的指标：
    features_num=19
    feature_cut=5
    # todo：每个模型里面增加调参模块，为贝叶斯网络增加保存最佳结构的代码
    random_state = 233

    global startTime, hour, log_path
    startTime = datetime.datetime.now().strftime('%m-%d')
    hour = datetime.datetime.now().strftime('%H_')
    log_path = os.path.join(filePath, 'log', fileName.split('.')[0], startTime)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # 删除缺失值>missingRage的变量
    del_file = del_missing(missing_rate=missingRate,
                           file_path=filePath,
                           file_name=fileName)

    # 读取删除缺失数据较多的特征之后的表
    dat = read_data(file=del_file)

    # 进行缺失值探查，并且将缺失值较多的，并进行缺失值填补
    dat_data = missing_dect(dat, missingRate=missingRate, method='mode')

    # 分箱 & 变量数值化
    dat_data = Boxing_Numeric(dat_data,cut=feature_cut)

    # 变量筛选
    xDF,yDF=MIC_selection(dat_data,k=features_num)


    # 划分测试集和训练集
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    kmodel = {'TPR_MEAN': [], 'TPR_STD': [], 'OPTS': [], 'L': []}

    ############################################## Logistic Model  #################################
    cols = {'A':[],'P': [], 'R': [], 'F1': [], 'AUC': [],'Threshold':[]}
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    opts=[]
    LRbest = {'name': -1, 'ypre': [], 'yprob': [], 'AUC': 0, 'ytest': [], 'yindex': []}
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    # for model in modelList:
    i = 1
    for train_index, test_index in KFold:
        xtrain = xDF.iloc[train_index, :]
        ytrain = yDF.iloc[train_index]
        xtest = xDF.iloc[test_index, :]
        ytest = yDF.iloc[test_index]

        lr = LogisticRegression(max_iter=1000,class_weight='auto').fit(np.array(xtrain), np.array(ytrain))
        ypre = lr.predict(np.array(xtest))
        yprob = lr.predict_proba(np.array(xtest))[:, 1]

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = Find_Optimal_Cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        a, p, r, f1, AUC = PRF1(np.array(ytest), ypre, yprob,threshold=opt[0])
        cols['A'].append(a)
        cols['P'].append(p)
        cols['R'].append(r)
        cols['F1'].append(f1)
        cols['AUC'].append(AUC)
        cols['Threshold'].append(opt[0])
        print(p, r, f1, AUC,opt[0])

        if AUC > LRbest['AUC']:
            LRbest['AUC'] = AUC
            LRbest['name'] = i
            LRbest['ypre'] = ypre
            LRbest['yprob'] = yprob
            LRbest['ytest'] = ytest.tolist()
            LRbest['yindex'] = ytest.index.tolist()

        i += 1

    kmodel['TPR_MEAN'].append(np.mean(tprs, axis=0))
    kmodel['TPR_STD'].append(np.std(tprs, axis=0))
    kmodel['L'].append('Logistic Regression')
    pd.DataFrame(cols).to_csv(os.path.join(log_path, hour) + 'result_LR.csv')
    # with open(os.path.join(log_path,hour)+'best_result_LR.json','w') as f:
    #     f.write(json.dumps(LRbest))


    ################################################# Random Forest  ####################################
    cols = {'A':[],'P': [], 'R': [], 'F1': [], 'AUC': [],'Threshold':[]}
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    opts = []
    RFbest = {'name': -1, 'ypre': [], 'yprob': [], 'AUC': 0, 'ytest': [], 'yindex': []}
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    i = 1
    for train_index, test_index in KFold:
        xtrain = xDF.iloc[train_index, :]
        ytrain = yDF.iloc[train_index]
        xtest = xDF.iloc[test_index, :]
        ytest = yDF.iloc[test_index]

        rf = RandomForestClassifier(n_estimators=100,
                                    random_state=random_state).fit(np.array(xtrain), np.array(ytrain))
        ypre = rf.predict(np.array(xtest))
        yprob = rf.predict_proba(np.array(xtest))[:, 1]

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = Find_Optimal_Cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        a, p, r, f1, AUC = PRF1(np.array(ytest), ypre, yprob, threshold=opt[0])
        cols['A'].append(a)
        cols['P'].append(p)
        cols['R'].append(r)
        cols['F1'].append(f1)
        cols['AUC'].append(AUC)
        cols['Threshold'].append(opt[0])
        print(p, r, f1, AUC, opt[0])

        if AUC > RFbest['AUC']:
            RFbest['AUC'] = AUC
            RFbest['name'] = i
            RFbest['ypre'] = ypre
            RFbest['yprob'] = yprob
            RFbest['ytest'] = ytest.tolist()
            RFbest['yindex'] = ytest.index.tolist()
        i += 1
    kmodel['TPR_MEAN'].append(np.mean(tprs, axis=0))
    kmodel['TPR_STD'].append(np.std(tprs, axis=0))
    kmodel['L'].append('Random Forest')
    pd.DataFrame(cols).to_csv(os.path.join(log_path, hour) + 'result_RF.csv')
    # with open(os.path.join(log_path,hour)+'best_result_RF.json','w') as f:
    #     f.write(json.dumps(RFbest))


    ##################################################### Bayesian Network  ######################################
    cols = {'A':[], 'P': [], 'R': [], 'F1': [], 'AUC': [],'Threshold':[]}
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    opts = []
    BNbest = {'name': -1, 'ypre': [], 'yprob': [], 'AUC': 0, 'ytest': [], 'model': '', 'yindex': []}
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    # for model in modelList:
    i = 1
    for train_index, test_index in KFold:
        xtrain = xDF.iloc[train_index, :]
        ytrain = yDF.iloc[train_index]
        xtest = xDF.iloc[test_index, :]
        ytest = yDF.iloc[test_index]

        model, net = Bmodel(xtrain, ytrain, idnum=i)
        ypre, yprob, errorList = Bpre(model, xtest, targetY)

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = Find_Optimal_Cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        a, p, r, f1, AUC = PRF1(np.array(ytest), ypre, yprob, threshold=opt[0])
        cols['A'].append(a)
        cols['P'].append(p)
        cols['R'].append(r)
        cols['F1'].append(f1)
        cols['AUC'].append(AUC)
        cols['Threshold'].append(opt[0])
        print(a, p, r, f1, AUC, opt[0])

        i += 1
        if AUC > BNbest['AUC']:
            BNbest['AUC'] = AUC
            BNbest['name'] = i
            BNbest['ypre'] = ypre
            BNbest['yprob'] = yprob
            BNbest['yindex'] = ytest.index.tolist()
            BNbest['ytest'] = ytest.tolist()
            BNbest['model'] = list(net.edges())

    kmodel['TPR_MEAN'].append(np.mean(tprs, axis=0))
    kmodel['TPR_STD'].append(np.std(tprs, axis=0))
    kmodel['L'].append('Bayesian Network')
    pd.DataFrame(cols).to_csv(os.path.join(log_path, hour) + 'resultBN.csv')
    # with open(os.path.join(log_path,hour)+'best_result_BN.json','w') as f:
    #     f.write(json.dumps(BNbest))

    ############################################# SVM Model  #################################
    from sklearn import svm
    cols = {'A':[],'P': [], 'R': [], 'F1': [], 'AUC': [],'Threshold':[]}
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    opts=[]
    SVMbest = {'name': -1, 'ypre': [], 'yprob': [], 'AUC': 0, 'ytest': [], 'yindex': []}
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    # for model in modelList:
    i = 1
    for train_index, test_index in tqdm(KFold):
        xtrain = xDF.iloc[train_index, :]
        ytrain = yDF.iloc[train_index]
        xtest = xDF.iloc[test_index, :]
        ytest = yDF.iloc[test_index]

        model = svm.SVC(kernel='rbf',gamma='auto',C=1,probability=True).fit(np.array(xtrain), np.array(ytrain))
        ypre = model.predict(np.array(xtest))
        yprob = model.predict_proba(np.array(xtest))[:, 1]

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = Find_Optimal_Cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        a, p, r, f1, AUC = PRF1(np.array(ytest), ypre, yprob,threshold=opt[0])
        cols['A'].append(a)
        cols['P'].append(p)
        cols['R'].append(r)
        cols['F1'].append(f1)
        cols['AUC'].append(AUC)
        cols['Threshold'].append(opt[0])
        print(a, p, r, f1, AUC,opt[0])

        if AUC > SVMbest['AUC']:
            SVMbest['AUC'] = AUC
            SVMbest['name'] = i
            SVMbest['ypre'] = ypre
            SVMbest['yprob'] = yprob
            SVMbest['ytest'] = ytest.tolist()
            SVMbest['yindex'] = ytest.index.tolist()

        i += 1
    kmodel['TPR_MEAN'].append(np.mean(tprs, axis=0))
    kmodel['TPR_STD'].append(np.std(tprs, axis=0))
    kmodel['L'].append('Support Vector Machine Classification')

    # plot_ROC_kfold(tprs, opts, l='Support Vector Machine Classification')
    pd.DataFrame(cols).to_csv(os.path.join(log_path, hour) + 'result_SVM.csv')
    # with open(os.path.join(log_path,hour)+'best_result_LR.json','w') as f:
    #     f.write(json.dumps(LRbest))

    ############################################## XGBoost Model  #################################
    from xgboost import XGBRFClassifier as XGBC
    from sklearn.metrics import mean_squared_error as MSE
    cols = {'A':[],'P': [], 'R': [], 'F1': [], 'AUC': [],'Threshold':[]}
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    opts=[]
    XGBbest = {'name': -1, 'ypre': [], 'yprob': [], 'AUC': 0, 'ytest': [], 'yindex': []}
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    # for model in modelList:
    i = 1
    for train_index, test_index in tqdm(KFold):
        xtrain = xDF.iloc[train_index, :]
        ytrain = yDF.iloc[train_index]
        xtest = xDF.iloc[test_index, :]
        ytest = yDF.iloc[test_index]

        model = XGBC(n_estimators =100,
                         random_state=random_state,
                         learning_rate=0.1,
                         booster='gbtree',
                         objective='reg:logistic',
                         silent=False).fit(np.array(xtrain), np.array(ytrain))
        ypre = model.predict(np.array(xtest))
        yprob = model.predict_proba(np.array(xtest))[:, 1]

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = Find_Optimal_Cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        a, p, r, f1, AUC = PRF1(np.array(ytest), ypre, yprob,threshold=opt[0])
        cols['A'].append(a)
        cols['P'].append(p)
        cols['R'].append(r)
        cols['F1'].append(f1)
        cols['AUC'].append(AUC)
        cols['Threshold'].append(opt[0])
        print(a, p, r, f1, AUC,opt[0])

        if AUC > XGBbest['AUC']:
            XGBbest['AUC'] = AUC
            XGBbest['name'] = i
            XGBbest['ypre'] = ypre
            XGBbest['yprob'] = yprob
            XGBbest['ytest'] = ytest.tolist()
            XGBbest['yindex'] = ytest.index.tolist()

        i += 1
    kmodel['TPR_MEAN'].append(np.mean(tprs, axis=0))
    kmodel['TPR_STD'].append(np.std(tprs, axis=0))
    kmodel['L'].append('XGBoost Classification')
    # plot_ROC_kfold(tprs, opts, l='XGBoost Classification')
    pd.DataFrame(cols).to_csv(os.path.join(log_path, hour) + 'result_XGB.csv')
    # with open(os.path.join(log_path,hour)+'best_result_LR.json','w') as f:
    #     f.write(json.dumps(LRbest))

    ############################################## LightGBM Model  #################################
    from lightgbm import LGBMClassifier as LGBMC
    from sklearn.metrics import mean_squared_error as MSE

    cols = {'A': [], 'P': [], 'R': [], 'F1': [], 'AUC': [], 'Threshold': []}
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    opts = []
    LGBMbest = {'name': -1, 'ypre': [], 'yprob': [], 'AUC': 0, 'ytest': [], 'yindex': []}
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    # for model in modelList:
    i = 1
    for train_index, test_index in tqdm(KFold):
        xtrain = xDF.iloc[train_index, :]
        ytrain = yDF.iloc[train_index]
        xtest = xDF.iloc[test_index, :]
        ytest = yDF.iloc[test_index]

        model = LGBMC(num_leaves=31,
                      learning_rate=0.05,
                      n_estimators=20).fit(np.array(xtrain), np.array(ytrain))
        ypre = model.predict(np.array(xtest))
        yprob = model.predict_proba(np.array(xtest))[:, 1]

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = Find_Optimal_Cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        # # 特征重要度
        # print('Feature importances:', list(gbm.feature_importances_))

        a, p, r, f1, AUC = PRF1(np.array(ytest), ypre, yprob, threshold=opt[0])
        cols['A'].append(a)
        cols['P'].append(p)
        cols['R'].append(r)
        cols['F1'].append(f1)
        cols['AUC'].append(AUC)
        cols['Threshold'].append(opt[0])
        print(a, p, r, f1, AUC, opt[0])

        if AUC > LGBMbest['AUC']:
            LGBMbest['AUC'] = AUC
            LGBMbest['name'] = i
            LGBMbest['ypre'] = ypre
            LGBMbest['yprob'] = yprob
            LGBMbest['ytest'] = ytest.tolist()
            LGBMbest['yindex'] = ytest.index.tolist()

        i += 1
    kmodel['TPR_MEAN'].append(np.mean(tprs, axis=0))
    kmodel['TPR_STD'].append(np.std(tprs, axis=0))
    kmodel['L'].append('LightGBM Classification')

    # plot_ROC_kfold(tprs, opts, l='LightGBM Classification')
    pd.DataFrame(cols).to_csv(os.path.join(log_path, hour) + 'result_LGBM.csv')
    # with open(os.path.join(log_path,hour)+'best_result_LR.json','w') as f:
    #     f.write(json.dumps(LRbest))

    plot_ROC_kmodel(kmodel)