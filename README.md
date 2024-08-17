# Machine_Learning
These codes represent 12 machine learning models: RandomForestClassifier, DecisionTreeClassifier, XGBClassifier, HistGradientBoostingClassifier,
GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, SVC, LogisticRegression, LinearDiscriminantAnalysis, KNeighborsClassifier, GaussianNB
Each of these models is run on 12 variable selection techniques: Lasso, LSVC, Tree Based Feature, RFECV, RELIEF, SURF, SURFSTAR, MULTISURF, MULTISURFSTAR,
SFS, ANOVA, MUTUAL INFORMATION. The pipeline contains: the variable selection method, the SVMSMOTE oversampling technique (for the case of imbalance data) and the model.
GridSearchCV and RandomizedSearchCV were used for hyperparameter optimization.
