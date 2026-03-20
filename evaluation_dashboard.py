from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix

classes_arr = sorted(y_test.unique())
n_classes   = len(classes_arr)
y_test_bin  = label_binarize(y_test, classes=classes_arr)
plot_colors = [PALETTE['primary'], PALETTE['success'], PALETTE['warning']]

fig = plt.figure(figsize=(20, 6))
gs  = gridspec.GridSpec(1, 3, figure=fig)
fig.suptitle('Model Evaluation Dashboard', fontsize=18)

ax1 = fig.add_subplot(gs[0])
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
print("✅ Saved!")
