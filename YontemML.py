#Python OOP yapısı kullanılmıştır.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
class YontemML:
  def __init__(self, X,y) :
      self.X = X
      self.y=  y
      self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.25,random_state=42)

  def Logistic(self):
    print("*****Logistic Regression*****")
    loj_model=LogisticRegression(random_state=42).fit(self.X_train,self.y_train)
    y_pred_loj=loj_model.predict(self.X_test)
    loj_skor= accuracy_score(y_pred_loj,self.y_test)
    loj_roc_auc=roc_auc_score(y_pred_loj,self.y_test)
    print("Logistic Regression ROC_AUC:",loj_roc_auc)
    print('Model Accuracy:', loj_skor)
    print("Classification report")
    cf_matrix_loj=confusion_matrix(y_pred_loj,self.y_test)
    sns.heatmap(cf_matrix_loj,annot=True,cbar=False, fmt='g')
    plt.title("Model Accuracy:"+str(loj_skor))
    plt.show()
    print(classification_report(y_pred_loj,self.y_test))
    return loj_model


  def Knn(self):
    print("*****KNN Algorithm*****")
    knn_model=KNeighborsClassifier().fit(self.X_train,self.y_train)
    y_pred_knn=knn_model.predict(self.X_test)
    knn_skor= accuracy_score(y_pred_knn,self.y_test)
    knn_roc_auc=roc_auc_score(y_pred_knn,self.y_test)
    print("KNN ROC_AUC:",knn_roc_auc)
    print('Model Accuracy:', knn_skor)
    print("Classification report")
    cf_matrix_knn=confusion_matrix(y_pred_knn,self.y_test)
    sns.heatmap(cf_matrix_knn,annot=True,cbar=False, fmt='g')
    plt.title("Model Accuracy:"+str(knn_skor))
    plt.show()
    print(classification_report(y_pred_knn,self.y_test))
    return knn_model


  def Svm(self):
    print("*****SVM Algorithm*****")
    svm_model=SVC(random_state=42).fit(self.X_train,self.y_train)
    y_pred_svm=svm_model.predict(self.X_test)
    svm_skor= accuracy_score(y_pred_svm,self.y_test)
    print('Model Accuracy:', svm_skor)
    print("Classification report")
    cf_matrix_svm=confusion_matrix(y_pred_svm,self.y_test)
    sns.heatmap(cf_matrix_svm,annot=True,cbar=False, fmt='g')
    plt.title("Model Accuracy:"+str(svm_skor))
    plt.show()
    print(classification_report(y_pred_svm,self.y_test))
    return svm_model

  def NaiveBayes(self):
    print("*****Naive Bayes Algorithm*****")
    nb_model=GaussianNB().fit(self.X_train,self.y_train)
    y_pred_nb=nb_model.predict(self.X_test)
    nb_skor= accuracy_score(y_pred_nb,self.y_test)
    nb_roc_auc=roc_auc_score(y_pred_nb,self.y_test)
    print("Naive Bayes ROC_AUC:",nb_roc_auc)
    print('Model Accuracy:', nb_skor)
    print("Classification report")
    cf_matrix_svm=confusion_matrix(y_pred_nb,self.y_test)
    sns.heatmap(cf_matrix_svm,annot=True,cbar=False, fmt='g')
    plt.title("Model Accuracy:"+str(nb_skor))
    plt.show()
    print(classification_report(y_pred_nb,self.y_test))
    return nb_model
  def Decisiontree(self):
    print("*****Decision Tree Algorithm*****")
    cart_model=DecisionTreeClassifier().fit(self.X_train,self.y_train)
    y_pred_cart=cart_model.predict(self.X_test)
    cart_skor= accuracy_score(y_pred_cart,self.y_test)
    dt_roc_auc=roc_auc_score(y_pred_cart,self.y_test)
    print("Decision tree ROC_AUC:",dt_roc_auc)
    print('Model Accuracy:', cart_skor)
    print("Classification report")
    cf_matrix_cart=confusion_matrix(y_pred_cart,self.y_test)
    sns.heatmap(cf_matrix_cart,annot=True,cbar=False, fmt='g')
    plt.title("Model Accuracy:"+str(cart_skor))
    plt.show()
    print(classification_report(y_pred_cart,self.y_test))
    return cart_model


  def Randomforest(self):
    print("*****Random Forest Algorithm*****")
    rf_model=RandomForestClassifier().fit(self.X_train,self.y_train)
    y_pred_rf=rf_model.predict(self.X_test)
    rf_skor= accuracy_score(y_pred_rf,self.y_test)
    rf_roc_auc=roc_auc_score(y_pred_rf,self.y_test)
    print("Random Forest ROC_AUC:",rf_roc_auc)
    print('Model Accuracy:', rf_skor)
    print("Classification report")
    cf_matrix_rf=confusion_matrix(y_pred_rf,self.y_test)
    sns.heatmap(cf_matrix_rf,annot=True,cbar=False, fmt='g')
    plt.title("Model Accuracy:"+str(rf_skor))
    plt.show()
    print(classification_report(y_pred_rf,self.y_test))
    return rf_model


  def Xgboost(self):
    print("*****XGBoost Algorithm*****")
    xgbm_model=XGBClassifier().fit(self.X_train,self.y_train)
    y_pred_xgbm=xgbm_model.predict(self.X_test)
    xgbm_skor= accuracy_score(y_pred_xgbm,self.y_test)
    xgbm_roc_auc=roc_auc_score(y_pred_xgbm,self.y_test)
    print("XGBM ROC_AUC:",xgbm_roc_auc)
    print('Model Accuracy:', xgbm_skor)
    print("Classification report")
    cf_matrix_xgbm=confusion_matrix(y_pred_xgbm,self.y_test)
    sns.heatmap(cf_matrix_xgbm,annot=True,cbar=False, fmt='g')
    plt.title("Model Accuracy:"+str(xgbm_skor))
    plt.show()
    print(classification_report(y_pred_xgbm,self.y_test))
    return xgbm_model


  def Gbm(self):
    print("*****GBM Algorithm*****")
    gbm_model=GradientBoostingClassifier().fit(self.X_train,self.y_train)
    y_pred_gbm=gbm_model.predict(self.X_test)
    gbm_skor= accuracy_score(y_pred_gbm,self.y_test)
    gbm_roc_auc=roc_auc_score(y_pred_gbm,self.y_test)
    print("GBM ROC_AUC:",gbm_roc_auc)
    print('Model Accuracy:', gbm_skor)
    print("Classification report")
    cf_matrix_gbm=confusion_matrix(y_pred_gbm,self.y_test)
    sns.heatmap(cf_matrix_gbm,annot=True,cbar=False, fmt='g')
    plt.title("Model Accuracy:"+str(gbm_skor))
    plt.show()
    print(classification_report(y_pred_gbm,self.y_test))
    return gbm_model

  def LightGBM(self):
    print("*****Light GBM Algorithm*****")
    lgbm_model=LGBMClassifier().fit(self.X_train,self.y_train)
    y_pred_lgbm=lgbm_model.predict(self.X_test)
    lgbm_skor= accuracy_score(y_pred_lgbm,self.y_test)
    lgbm_roc_auc=roc_auc_score(y_pred_lgbm,self.y_test)
    print("Light GBM ROC_AUC:",lgbm_roc_auc)
    print('Model Accuracy:', lgbm_skor)
    print("Classification report")
    cf_matrix_lgbm=confusion_matrix(y_pred_lgbm,self.y_test)
    sns.heatmap(cf_matrix_lgbm,annot=True,cbar=False, fmt='g')
    plt.title("Model Accuracy:"+str(lgbm_skor))
    plt.show()
    print(classification_report(y_pred_lgbm,self.y_test))
    return lgbm_model


  def Mlpc(self):
    print("*****MLPC Algorithm*****")
    mlpc_model=MLPClassifier().fit(self.X_train,self.y_train)
    y_pred_mlpc=mlpc_model.predict(self.X_test)
    mlpc_skor= accuracy_score(y_pred_mlpc,self.y_test)
    mlpc_roc_auc=roc_auc_score(y_pred_mlpc,self.y_test)
    print("Light GBM ROC_AUC:",mlpc_roc_auc)
    print('Model Accuracy:', mlpc_skor)
    print("Classification report")
    cf_matrix_mlpc=confusion_matrix(y_pred_mlpc,self.y_test)
    sns.heatmap(cf_matrix_mlpc,annot=True,cbar=False, fmt='g')
    plt.title("Model Accuracy:"+str(mlpc_skor))
    plt.show()
    print(classification_report(y_pred_mlpc,self.y_test))
    return mlpc_model
