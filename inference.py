def predict_resident_city(new_data, xgb_model, tf_model, lr_model, label_encoder):
    """
    Predict resident city for new data.
    
    Args:
        new_data (pd.DataFrame): New input data with columns ['user_id', 'avg_purchase_amount', 'weekly_activity'].
        xgb_model: Trained XGBoost model.
        tf_model: Trained TensorFlow embedding model.
        lr_model: Trained Logistic Regression meta-learner.
        label_encoder: LabelEncoder used for target during training.
        
    Returns:
        Predicted city labels (decoded from numeric to original names).
    """
    # Step 1: Extract features
    user_id = new_data['user_id'].values
    xgb_features = new_data[['avg_purchase_amount', 'weekly_activity']]
    
    # Step 2: Generate XGBoost probabilities
    xgb_probs = xgb_model.predict_proba(xgb_features)
    
    # Step 3: Generate TensorFlow probabilities
    tf_probs = tf_model.predict(user_id)
    
    # Step 4: Stack predictions for meta-learner
    stacked_probs = np.hstack([xgb_probs, tf_probs])
    
    # Step 5: Predict with Logistic Regression
    final_preds = lr_model.predict(stacked_probs)
    
    # Step 6: Decode city labels
    decoded_cities = label_encoder.inverse_transform(final_preds)
    
    return decoded_cities

# Example Usage
new_users = pd.DataFrame({
    'user_id': [123, 456, 789],  # Must be within 0–9999 (as in training)
    'avg_purchase_amount': [45.6, 60.2, 30.8],
    'weekly_activity': [75, 20, 90]
})

# Load pre-trained models and label_encoder (assumes they are saved)
# xgb_model = xgb.Booster()
# xgb_model.load_model('xgb_model.json')
# tf_model = tf.keras.models.load_model('tf_embedding_model.h5')
# lr_model = joblib.load('lr_meta_model.pkl')
# label_encoder = joblib.load('label_encoder.pkl')

predicted_cities = predict_resident_city(new_users, xgb_model, tf_model, lr_model, label_encoder)
print("Predicted Cities:", predicted_cities)