os.makedirs('model_artifacts', exist_ok=True)

# Save all artifacts
joblib.dump(best_lr,      'model_artifacts/logistic_regression_best.pkl')
joblib.dump(preprocessor, 'model_artifacts/preprocessor.pkl')
joblib.dump(rfe,          'model_artifacts/rfe_selector.pkl')

# Save metrics
metrics_out = {
    'accuracy':  float(accuracy_score(y_test, y_pred_best)),
    'f1':        float(f1_score(y_test, y_pred_best, average='weighted')),
    'roc_auc':   float(roc_auc_score(y_test, y_prob_best, multi_class='ovr', average='weighted')),
    'saved_at':  datetime.now().isoformat()
}
with open('model_artifacts/metrics.json', 'w') as f:
    json.dump(metrics_out, f, indent=2)

print("✅ All artifacts saved!")
print(os.listdir('model_artifacts'))
