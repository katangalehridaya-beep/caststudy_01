def evaluate_model(model, X, y, split_name='Test'):
    y_pred       = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    print(f'\n{"="*55}')
    print(f'  📊 {split_name} Evaluation')
    print(f'{"="*55}')
    print(f'  Accuracy         : {accuracy_score(y, y_pred):.4f}')
    print(f'  Balanced Accuracy: {balanced_accuracy_score(y, y_pred):.4f}')
    print(f'  F1 (weighted)    : {f1_score(y, y_pred, average="weighted"):.4f}')
    print(f'  ROC-AUC (ovr)    : {roc_auc_score(y, y_pred_proba, multi_class="ovr", average="weighted"):.4f}')
    print(f'  Log Loss         : {log_loss(y, y_pred_proba):.4f}')
    print(f'  MCC              : {matthews_corrcoef(y, y_pred):.4f}')
    print(f'  Cohen Kappa      : {cohen_kappa_score(y, y_pred):.4f}')
    print(f'\n{classification_report(y, y_pred)}')

    return y_pred, y_pred_proba

y_pred_val,  y_prob_val  = evaluate_model(lr_model, X_val_sel,  y_val,  'Validation')
y_pred_test, y_prob_test = evaluate_model(lr_model, X_test_sel, y_test, 'Test')
