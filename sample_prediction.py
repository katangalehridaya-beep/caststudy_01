# Sample prediction test
sample     = X_test.iloc[[0]]
sample_sc  = preprocessor.transform(sample)
sample_sel = rfe.transform(sample_sc)
pred_proba = best_lr.predict_proba(sample_sel)[0]
pred_class = best_lr.predict(sample_sel)[0]

print("─"*40)
print(f"  Sample Prediction")
print("─"*40)
for i, p in enumerate(pred_proba):
    print(f"  Class {i}: {p*100:.1f}%")
print(f"\n  Predicted Class : {pred_class}")
print(f"  Actual Class    : {y_test.iloc[0]}")
print("─"*40)
