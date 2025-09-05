import os
import shap
import pandas as pd
import joblib   # use joblib instead of pickle for sklearn/xgboost models
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid

# ----------------------
# Load model (only the XGBClassifier was saved)
# ----------------------
model = joblib.load("best_xgb_model.pkl")

# Reconstruct feature names (must match training)
# ⚠️ Replace with your actual training DataFrame column names
feature_names = list(model.get_booster().feature_names)

# SHAP explainer
explainer = shap.Explainer(model)

def predict_with_explanation(input_data: pd.DataFrame, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    # Ensure input data columns align with training
    input_data = input_data[feature_names]
    predictions = model.predict(input_data)
    proba_all = model.predict_proba(input_data)  # [[p0, p1], ...]

    shap_values_all = explainer(input_data)
    explanations = []

    for i in range(len(input_data)):
        shap_exp = shap_values_all[i]
        shap_contribs = np.array(shap_exp.values).flatten()
        feature_vals = np.array(shap_exp.data).flatten()
        base_value = shap_exp.base_values

        # Compute f(x) and Ef(x) in probability terms
        fx_logodds = shap_contribs.sum() + base_value
        fx_prob = expit(fx_logodds)
        base_prob = expit(base_value)

        # Top 5 contributing features
        top_indices = np.argsort(np.abs(shap_contribs))[-5:][::-1]
        top_shap_values = shap_contribs[top_indices]
        top_feature_values = feature_vals[top_indices]
        top_feature_names = [feature_names[j] for j in top_indices]

        # ---------------------------
        # Pick label + probability
        # ---------------------------
        failure_prob = proba_all[i, 1]
        normal_prob = proba_all[i, 0]

        if predictions[i] == 1:  # FAILURE
            pred_label = "FAILURE"
            pred_prob = failure_prob
        else:  # NORMAL
            pred_label = "NORMAL"
            pred_prob = normal_prob

        explanation = (
            f"Row {i}: The model predicts **{pred_label}** "
            f"with a probability of **{pred_prob:.2f}** "
            f"(approx. f(x) = {fx_prob:.3f}, E[f(x)] = {base_prob:.3f}).\n\n"
        )

        # Warn if system is trending toward failure
        if pred_label == "NORMAL" and failure_prob >= 0.40:
            explanation += (
                f"⚠️ Warning: Although predicted NORMAL, the failure probability is **{failure_prob:.2f}**. "
                f"This suggests the system may be heading towards a failure state soon.\n\n"
            )

        explanation += (
            "The following features contributed most to this failure prediction:\n"
            if pred_label == "FAILURE" else
            "The following features helped predict a normal condition:\n"
        )

        # ---------------------------
        # Recommendations
        # ---------------------------
        recommendations = []
        for rank, (fname, sval, fval) in enumerate(zip(top_feature_names, top_shap_values, top_feature_values), 1):
            direction = "increased" if sval > 0 else "decreased"
            effect = "risk of failure" if sval > 0 else "chance of normal operation"
            explanation += (
                f"{rank}. **{fname}** (value = {fval:.2f}) {direction} the model's prediction "
                f"towards **{effect}** by **{abs(sval):.3f}**.\n"
            )

            # 🎯 Recommendation logic
            if pred_label == "FAILURE":
                if sval > 0:  # pushing toward failure
                    if fval > 70:
                        recommendations.append(f"• Reduce **{fname}** slightly. Its high value ({fval:.2f}) is contributing to failure.")
                    elif fval < 30:
                        recommendations.append(f"• Increase **{fname}** slightly. Its low value ({fval:.2f}) is destabilizing performance.")
                else:  # helping normal
                    recommendations.append(f"• Maintain current level of **{fname}** ({fval:.2f}). It supports normal operation.")
            else:  # NORMAL prediction
                if sval < 0:
                    if fval < 25:
                        recommendations.append(f"• Consider increasing **{fname}**. It's quite low ({fval:.2f}) and important for reliability.")
                    elif fval > 75:
                        recommendations.append(f"• Consider reducing **{fname}**. It's very high ({fval:.2f}), and although helping, could destabilize.")
                elif sval > 0:
                    if fval > 70:
                        recommendations.append(f"• Monitor **{fname}**. It's high ({fval:.2f}) and contributes slightly toward risk.")
                    elif fval < 20:
                        recommendations.append(f"• Monitor **{fname}**. It's very low ({fval:.2f}) and increases risk.")

        explanation += (
            f"\nThe feature **{top_feature_names[0]}** had the most significant influence. "
            f"Its value ({top_feature_values[0]:.2f}) strongly indicates a **{pred_label.lower()}**.\n"
        )

        if recommendations:
            explanation += "\n### 🛠️ Recommendations\n" + "\n".join(recommendations)

        # 📊 Custom horizontal bar plot
        fig, ax = plt.subplots(figsize=(4.8, 2.8))
        colors = ["green" if val < 0 else "red" for val in top_shap_values]

        ax.barh(top_feature_names[::-1], top_shap_values[::-1], color=colors[::-1])
        ax.set_xlabel("SHAP Value", fontsize=8)
        ax.set_title("Top 5 Feature Impacts", fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=7)
        plt.tight_layout(pad=0.5)

        plot_path = os.path.join(output_dir, f"shap_custom_row_{i}.png")
        fig.savefig(plot_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

        explanations.append((explanation, plot_path))

    # Return predictions + probability of predicted class
    pred_probs = [proba_all[j, predictions[j]] for j in range(len(predictions))]
    return predictions, pred_probs, explanations
