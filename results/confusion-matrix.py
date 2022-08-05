# Confusion Matrix
y_pred = np.array(list(map(lambda x: np.argmax(x), model.predict(x_test))))

cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred, target_names=label_mapping.keys())

plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap ='Spectral_r')
plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----\n", clr)