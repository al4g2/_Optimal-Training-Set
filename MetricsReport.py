from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt


def metrics_report(y_true, y_pred, title):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f'\nConfusion matrix, {title}: \n %s' % cm)

    # Compute Precision, recall, F-measure and support
    print(f'\nClassification Report, {title}: \n %s' % classification_report(y_true, y_pred))

    # ROC and FPRvsTRP
    roc_auc = roc_auc_score(y_true, y_pred)
    print(f'\nROC-AUC: ', roc_auc)
    # fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # f1-score
    f1 = f1_score(y_true, y_pred)
    print('f1-score: ', f1)

    # # Plot ROC Curve
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.3f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'Receiver Operating Characteristic \n {title}', wrap=True)
    # plt.legend(loc="lower right")
    # plt.show()
