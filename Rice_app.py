
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataDimensionalityWarning
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

for col in df:
  if df[col].dtype =='object':
    df[col]=OrdinalEncoder().fit_transform(df[col].values.reshape(-1,1))
df

class_label =df['CLASS']
df = df.drop(['CLASS'], axis =1)
df = (df-df.min())/(df.max()-df.min())
df['CLASS']=class_label
df


rice_data = df.copy()
le = preprocessing.LabelEncoder()
area = le.fit_transform(list(rice_data["AREA"]))
perimete = le.fit_transform(list(rice_data["PERIMETER"]))
major_axis = le.fit_transform(list(rice_data["MAJOR_AXIS"]))
minor_axis = le.fit_transform(list(rice_data["MINOR_AXIS"]))
eccentricity = le.fit_transform(list(rice_data["ECCENTRICITY"]))
eqdiasq = le.fit_transform(list(rice_data["EQDIASQ"]))
solidity = le.fit_transform(list(rice_data["SOLIDITY"]))
convex_area = le.fit_transform(list(rice_data["CONVEX_AREA"]))
extent = le.fit_transform(list(rice_data["EXTENT"]))
aspect_ratio = le.fit_transform(list(rice_data["ASPECT_RATIO"]))
roundness = le.fit_transform(list(rice_data["ROUNDNESS"]))
compactness = le.fit_transform(list(rice_data["COMPACTNESS"]))
shapefactor_1 = le.fit_transform(list(rice_data["SHAPEFACTOR_1"]))
shapefactor_2 = le.fit_transform(list(rice_data["SHAPEFACTOR_2"]))
shapefactor_3 = le.fit_transform(list(rice_data["SHAPEFACTOR_3"]))
shapefactor_4 = le.fit_transform(list(rice_data["SHAPEFACTOR_4"]))
Class = le.fit_transform(list(rice_data["CLASS"]))

x = list(zip(area, perimete, major_axis, minor_axis, eccentricity, eqdiasq, solidity, convex_area, extent, aspect_ratio, roundness, compactness, shapefactor_1, shapefactor_2, shapefactor_3, shapefactor_4))
y = list(Class)

num_folds = 5
seed = 7
scoring = 'accuracy'


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)

np.shape(x_train), np.shape(x_test)

models = []
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []

print("Performance on Training set")

for name, model in models:
	kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	msg += '\n'
	print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
dt = DecisionTreeClassifier()
nb = GaussianNB()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()

best_model = rf
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
print("Best Model Accuracy Score on Test Set:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

best_model.fit(x_train, y_train)
n_classes = len(set(y_train))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    y_test_bin = pd.Series(y_test).map(lambda x: 1 if x == i else 0)
    y_score_bin = best_model.predict_proba(x_test)[:, i]
    fpr[i], tpr[i], _ = roc_curve(y_test_bin, y_score_bin)
    roc_auc[i] = roc_auc_score(y_test_bin, y_score_bin)

plt.figure()
colors = ['blue', 'red', 'green', 'yellow', 'purple', 'cyan']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Multi-class')
plt.legend(loc="lower right")
plt.savefig('LOC_ROC')
plt.show()

for x in range(len(y_pred)):
	print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x],)