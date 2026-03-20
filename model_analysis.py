import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

num_feat = [c for c in df.select_dtypes(include='number').columns if c != TARGET]
n_cols = 3
n_rows = int(np.ceil(len(num_feat)/n_cols))

# 1. Univariate
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows*4))
fig.suptitle('Univariate Analysis', fontsize=16)
axes = axes.flatten()
for i, col in enumerate(num_feat):
    axes[i].hist(df[col], bins=30, color=PALETTE['primary'], alpha=0.7)
    axes[i].set_title(col)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig('/content/univariate.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ univariate.png saved")

# 2. Bivariate
classes = sorted(df[TARGET].unique())
colors  = [PALETTE['danger'], PALETTE['success'], PALETTE['warning']]
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows*4))
fig.suptitle('Bivariate Analysis', fontsize=16)
axes = axes.flatten()
for i, col in enumerate(num_feat):
    for cls, clr in zip(classes, colors):
        axes[i].hist(df[df[TARGET]==cls][col], bins=25,
                     alpha=0.5, color=clr, label=f'Class {cls}')
    axes[i].set_title(col)
    axes[i].legend(fontsize=7)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig('/content/bivariate.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ bivariate.png saved")

# 3. Mutual Info
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(df[num_feat], df[TARGET], random_state=42)
mi_df = pd.Series(mi, index=num_feat).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(mi_df.index, mi_df.values, color=PALETTE['primary'])
ax.set_title('Mutual Information Scores')
plt.tight_layout()
plt.savefig('/content/mutual_info.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ mutual_info.png saved")

# 4. Evaluation Dashboard
classes_arr = sorted(y_test.unique())
n_classes   = len(classes_arr)
y_test_bin  = label_binarize(y_test, classes=classes_arr)
plot_colors = [PALETTE['primary'], PALETTE['success'], PALETTE['warning']]

fig = plt.figure(figsize=(20, 6))
gs  = gridspec.GridSpec(1, 3, figure=fig)
fig.suptitle('Model Evaluation Dashboard', fontsize=18)

ax1 = fig.add_subplot(gs[0])
from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_test, y_pred_best)
im  = ax1.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax1)
for i in range(n_classes):
    for j in range(n_classes):
        ax1.text(j, i, f'{cm[i,j]}', ha='center', va='center',
                 fontsize=12, color='white' if cm[i,j] > cm.max()/2 else 'black')
ax1.set_xticks(range(n_classes))
ax1.set_yticks(range(n_classes))
ax1.set_xticklabels([f'Class {c}' for c in classes_arr])
ax1.set_yticklabels([f'Class {c}' for c in classes_arr])
ax1.set_title('Confusion Matrix')

ax2 = fig.add_subplot(gs[1])
for idx, cls in enumerate(classes_arr):
    fpr, tpr, _ = roc_curve(y_test_bin[:, idx], y_prob_best[:, idx])
    auc_val     = roc_auc_score(y_test_bin[:, idx], y_prob_best[:, idx])
    ax2.plot(fpr, tpr, color=plot_colors[idx], lw=2,
             label=f'Class {cls} (AUC={auc_val:.3f})')
ax2.plot([0,1],[0,1],'--', color='#8B949E')
ax2.set_title('ROC Curve')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[2])
for idx, cls in enumerate(classes_arr):
    prec, rec, _ = precision_recall_curve(y_test_bin[:, idx], y_prob_best[:, idx])
    ax3.plot(rec, prec, color=plot_colors[idx], lw=2, label=f'Class {cls}')
ax3.set_title('Precision-Recall Curve')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/evaluation_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ evaluation_dashboard.png saved")
